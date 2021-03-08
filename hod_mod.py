from halotools.sim_manager import CachedHaloCatalog, FakeSim
from halotools.empirical_models import PrebuiltHodModelFactory, Zheng07Cens, Zheng07Sats, TrivialPhaseSpace, NFWPhaseSpace, HodModelFactory
from halotools.mock_observables import return_xyz_formatted_array
import numpy as np
from warnings import simplefilter

#ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


class model():
    """get halo catalog and model instance"""
    def _get_sim(self,sim):
        if sim == "bolplanck":
            halocat = CachedHaloCatalog(fname = '/home/lom31/.astropy/cache/halotools/halo_catalogs/bolplanck/rockstar/hlist_1.00231.list.halotools_v0p4.hdf5',update_cached_fname = True)
            halocat.redshift=0.
        elif sim == "old":
            halocat = CachedHaloCatalog(fname = '/home/lom31/Halo/hlist_1.00231.list.halotoo/ls_v0p1.hdf5',update_cached_fname = True)
            halocat.redshift=0.
        elif sim == "smdpl":
            halocat = CachedHaloCatalog(fname = '/home/lom31/.astropy/cache/halotools/halo_catalogs/SMDPL/rockstar/2019-07-03-18-38-02-9731.dat.my_cosmosim_halos.hdf5',update_cached_fname = True)
            halocat.redshift=0.
        elif self['sim'] == "mdr1":
            halocat = CachedHaloCatalog(fname = '/home/lom31/.astropy/cache/halotools/halo_catalogs/multidark/rockstar/hlist_0.68215.list.halotools_v0p4.hdf5',update_cached_fname = True)

        return halocat

    def _get_model(self,param,sim):
        halocat = self._get_sim(sim)
        if param == 'mvir':
            cens_occ_model = Zheng07Cens()
            cens_prof_model = TrivialPhaseSpace()
            sats_occ_model =  Zheng07Sats(modulate_with_cenocc=True)
            sats_prof_model = NFWPhaseSpace()
        elif param == 'vmax':
            cens_occ_model = Zheng07Cens(prim_haloprop_key = 'halo_vmax')
            cens_prof_model = TrivialPhaseSpace()
            sats_occ_model = Zheng07Sats(prim_haloprop_key = 'halo_vmax',
                                         modulate_with_cenocc=True)
            sats_prof_model = NFWPhaseSpace()
        
        model_instance = HodModelFactory(centrals_occupation = cens_occ_model,
                                         centrals_profile = cens_prof_model,
                                         satellites_occupation = sats_occ_model,
                                         satellites_profile = sats_prof_model)
        try:
            model_instance.mock.populate()
        except:
            model_instance.populate_mock(halocat)
        
        return model_instance
