import unittest
import numpy as np
from wildfire.model import Config, Dataset, FirePathEnv, angular_alignment, diffusion_steps, emc_percent, iou, reward_table, synthetic
from wildfire.rollout import evaluate_observed,fast_spread,q_table,spread

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
    def test_ji_bonus_shapes_training_but_not_footprint(self):
        cfg=Config();data=synthetic(1,cfg);bonus=np.full((cfg.size,cfg.size),5.)
        plain=FirePathEnv(data,cfg);plain.reset();_,r0,_,i0=plain.step(2)
        shaped=FirePathEnv(data,cfg,bonus=bonus);shaped.reset();_,r1,_,i1=shaped.step(2)
        self.assertAlmostEqual(r1-r0,5.*len(i1['path']))
        self.assertEqual(i0['path'],i1['path'])
    def test_consumed_fuel_removes_stay_reward(self):
        cfg=Config(consumed_fuel=True);env=FirePathEnv(synthetic(1,cfg),cfg);env.reset()
        _,reward,_,info=env.step(8)
        self.assertEqual(reward,0.);self.assertTrue(all(r==0. for _,_,r in info['path']))
        cfg=Config();env=FirePathEnv(synthetic(1,cfg),cfg);env.reset()
        self.assertNotEqual(env.step(8)[1],0.)
    def test_moisture_offset_lowers_reward_and_matches_table(self):
        cfg=Config(ndvi_mode='raw');data=synthetic(1,cfg)
        wet=Dataset(dict(data.fields,moisture_offset_pct=np.full_like(data.fields['ndvi'],3.)),data.metadata)
        self.assertLess(wet.reward((7,10),2,0,cfg),data.reward((7,10),2,0,cfg))
        self.assertAlmostEqual(reward_table(wet,cfg)[0,10,7,2],wet.reward((7,10),2,0,cfg))
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
    def test_reward_table_matches_reward(self):
        for mode in ('raw','literal'):
            cfg=Config(ndvi_mode=mode);data=synthetic(3,cfg);table=reward_table(data,cfg)
            for t,x,y,a in [(0,6,10,2),(5,25,25,4),(99,49,0,8),(40,30,12,7)]:
                self.assertAlmostEqual(table[t,y,x,a],data.reward((x,y),a,t,cfg),places=5)
    def test_fast_spread_matches_spread(self):
        import torch
        from wildfire.experiment import QNetwork
        torch.manual_seed(0);net=QNetwork();cfg=Config(size=20,horizon=12);data=synthetic(4,cfg)
        slow=spread(data,cfg,net,'dqn',(5,5))
        fast=fast_spread(data,cfg,q_table(data,net),reward_table(data,cfg),(5,5),policy='dqn')
        np.testing.assert_array_equal(slow,fast)
        soft=fast_spread(data,cfg,q_table(data,net),reward_table(data,cfg),(5,5),seed=1,temperature=2.)
        self.assertTrue(np.all(soft[:-1]<=soft[1:]))
    def test_beyond_thesis_spread_options(self):
        import torch
        from wildfire.experiment import QNetwork
        torch.manual_seed(0);net=QNetwork();cfg=Config(size=20,horizon=12);data=synthetic(4,cfg)
        q,r=q_table(data,net),reward_table(data,cfg)
        base=fast_spread(data,cfg,q,r,(5,5),seed=1,temperature=4.)[-1]
        strict=fast_spread(data,cfg,q,r,(5,5),seed=1,temperature=4.,theta=float(r.max())+1)[-1]
        self.assertEqual(strict.sum(),1)
        wall=np.zeros((20,20),bool);wall[:,8]=True
        blocked=fast_spread(data,cfg,q,r,(5,5),seed=1,temperature=4.,barrier=wall)[-1]
        self.assertFalse(blocked[:,8:].any())
        hard=fast_spread(data,cfg,q,r,(5,5),seed=1,temperature=4.,sharpness=1e6)[-1]
        self.assertGreater(hard.sum(),1);self.assertLessEqual(base.sum(),400)
    def test_residence_time_stops_spread(self):
        import torch
        from wildfire.experiment import QNetwork
        torch.manual_seed(0);net=QNetwork();cfg=Config(size=20,horizon=12);data=synthetic(4,cfg)
        q,r=q_table(data,net),reward_table(data,cfg)
        free=fast_spread(data,cfg,q,r,(5,5),seed=1,temperature=4.)
        short=fast_spread(data,cfg,q,r,(5,5),seed=1,temperature=4.,residence=1)
        np.testing.assert_array_equal(free[0],short[0])
        self.assertLess(short[-1].sum(),free[-1].sum())
        np.testing.assert_array_equal(short[-2],short[-1])
    def test_fifth_weight_scores_conifer_share(self):
        cfg=Config(ndvi_mode='raw');data=synthetic(1,cfg)
        pine=Dataset(dict(data.fields,conifer_fraction=np.ones_like(data.fields['ndvi'])),data.metadata)
        five=Config(ndvi_mode='raw',weights=(.2,.1,.1,.4,.2))
        self.assertAlmostEqual(pine.reward((7,10),2,0,five)-pine.reward((7,10),2,0,Config(ndvi_mode='raw',weights=(.2,.1,.1,.4,0))),.2)
        self.assertAlmostEqual(reward_table(pine,five)[0,10,7,2],pine.reward((7,10),2,0,five),places=5)
        with self.assertRaises(ValueError):FirePathEnv(data,five)
    def test_spotting_jumps_downwind(self):
        import torch
        from wildfire.experiment import QNetwork
        torch.manual_seed(0);net=QNetwork();cfg=Config(size=30,horizon=6,diffusion_scale=.01);data=synthetic(0,cfg)
        q,r=q_table(data,net),reward_table(data,cfg)
        base=fast_spread(data,cfg,q,r,(5,15),seed=2,temperature=4.,residence=6)
        spot=fast_spread(data,cfg,q,r,(5,15),seed=2,temperature=4.,residence=6,spot=1.,spot_m=800.)
        ys,xs=np.nonzero(spot[-1]&~base[-1]);self.assertTrue(len(xs))
        self.assertGreater(np.max(np.hypot(xs-5,ys-15)),1.5)
    def test_wind_term_scales_with_speed(self):
        cfg=Config(ndvi_mode='raw');data=synthetic(1,cfg)
        calm=Dataset(dict(data.fields,wind_speed_ms=np.zeros_like(data.fields['wind_speed_ms'])),data.metadata)
        scaled=Config(ndvi_mode='raw',wind_ref_ms=5.);still=Config(ndvi_mode='raw',weights=cfg.weights[:3]+(0,))
        self.assertAlmostEqual(calm.reward((7,10),2,0,scaled),calm.reward((7,10),2,0,still))
        np.testing.assert_allclose(reward_table(data,scaled)[0,10,7],[data.reward((7,10),a,0,scaled) for a in range(9)],atol=1e-5)
    def test_micromet_downscaling(self):
        from data.synth_weather import downscale
        z=np.add.outer(np.zeros(50),np.linspace(0,500,50))
        temp,dew,u,v=downscale(z,250.,10.,-5.,5.,270.,1.,5)
        self.assertGreater(temp[:,0].mean(),temp[:,-1].mean());self.assertTrue(np.all(dew<=temp))
        self.assertAlmostEqual(float(np.hypot(u,v).mean()),5.,places=6)
        flat=downscale(z,250.,10.,-5.,5.,270.,0.,5)
        np.testing.assert_allclose(np.hypot(flat[2],flat[3]),5.)
    def test_weight_simplex_validation(self):
        cfg=Config(weights=(1,1,1,1))
        with self.assertRaises(ValueError):FirePathEnv(synthetic(1,cfg),cfg)

if __name__=='__main__':unittest.main()
