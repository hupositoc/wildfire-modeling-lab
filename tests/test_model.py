import unittest
import numpy as np
from wildfire.model import Config, Dataset, FirePathEnv, angular_alignment, diffusion_steps, emc_percent, iou, synthetic
from wildfire.rollout import evaluate_observed,spread

class ModelTests(unittest.TestCase):
    def test_emc_increases_with_humidity(self):
        self.assertGreater(emc_percent(20,.8),emc_percent(20,.2))
    def test_alignment_and_circular_wrap(self):
        self.assertAlmostEqual(angular_alignment(2,0),1)
        self.assertAlmostEqual(angular_alignment(6,0),-1)
        self.assertAlmostEqual(angular_alignment(2,360),1)
        self.assertEqual(angular_alignment(8,0),0)
    def test_rounding_is_explicit(self):
        cfg=Config();self.assertEqual(diffusion_steps(3.9,cfg),7)
        cfg.rounding='ceil';self.assertEqual(diffusion_steps(3.9,cfg),8)
    def test_no_wraparound(self):
        cfg=Config();env=FirePathEnv(synthetic(7,cfg),cfg,start=(1,10));env.reset()
        _,_,done,info=env.step(6)
        self.assertTrue(done);self.assertEqual(info['position'][0],0)
    def test_cannot_cross_nonfuel_barrier(self):
        cfg=Config();data=synthetic(1,cfg)
        data.fields['ndvi'][:,:,8]=-.5
        env=FirePathEnv(data,cfg,start=(6,10));env.reset();env.step(2)
        self.assertEqual(env.pos[0],7)
    def test_horizon_terminates(self):
        cfg=Config();env=FirePathEnv(synthetic(1,cfg),cfg);env.reset()
        for _ in range(cfg.horizon):_,_,done,_=env.step(8)
        self.assertTrue(done)
        with self.assertRaises(RuntimeError):env.step(8)
    def test_iou_is_spatial_overlap(self):
        self.assertAlmostEqual(iou([1,1,0],[0,1,1]),1/3)
        self.assertEqual(iou([1,0],[1,0]),1)
        self.assertTrue(np.isnan(iou([0],[0])))
    def test_synthetic_is_reproducible_and_labelled(self):
        a=synthetic(7,Config());b=synthetic(7,Config())
        self.assertEqual(a.metadata['kind'],'synthetic')
        for key in a.fields:np.testing.assert_array_equal(a.fields[key],b.fields[key])
    def test_literal_and_raw_ndvi_are_not_silently_conflated(self):
        data=synthetic(3,Config());pos=(6,10)
        a=data.reward(pos,2,0,Config(ndvi_mode='literal'))
        b=data.reward(pos,2,0,Config(ndvi_mode='raw'))
        self.assertNotEqual(a,b)
    def test_bad_inputs_rejected(self):
        data=synthetic(1,Config());data.fields['rh_fraction'][0,0,0]=50
        with self.assertRaises(ValueError):Dataset(data.fields,data.metadata)
    def test_synthetic_mask_cannot_be_called_observed(self):
        with self.assertRaises(ValueError):evaluate_observed([1],[1],{'kind':'synthetic'})
    def test_placeholder_provenance_is_rejected(self):
        meta={'kind':'observational','source_url':'REPLACE_WITH_SOURCE','event_id':'example','mask_method':'example','time_window':'example','grid_bounds':[0,0,1,1],'grid_origin':'lower-left','version':'example'}
        with self.assertRaises(ValueError):evaluate_observed([1],[1],meta)
    def test_completed_boundary_episode_cannot_step(self):
        cfg=Config();env=FirePathEnv(synthetic(7,cfg),cfg,start=(1,10));env.reset();env.step(6)
        with self.assertRaises(RuntimeError):env.step(2)
    def test_parallel_rollout_preserves_burned_cells(self):
        cfg=Config(size=8,horizon=3);data=synthetic(4,cfg)
        frames=spread(data,cfg,policy='random',start=(2,2),seed=4)
        self.assertEqual(frames.shape,(3,8,8))
        self.assertTrue(np.all(frames[:-1]<=frames[1:]))
    def test_weight_simplex_validation(self):
        cfg=Config(weights=(1,1,1,1))
        with self.assertRaises(ValueError):FirePathEnv(synthetic(1,cfg),cfg)

if __name__=='__main__':unittest.main()
