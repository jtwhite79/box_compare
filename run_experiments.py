import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import flopy
import pyemu

base_d = "template_org"
base_pst = "temp.pst"
new_d = "template_new"
new_pst = "new.pst"

forecasts = ["part_time","part_east"]

def add_extras_without_pps():

    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(base_d, new_d)

    m = flopy.modflow.Modflow.load("rect_model.nam", model_ws=new_d, verbose=True)

    with open(os.path.join(new_d, "wel.dat")) as f:
        lines = f.readlines()
    with open(os.path.join(new_d, 'wel.dat.tpl'), 'w') as f:
        f.write("ptf ~\n")
        [f.write(line) for line in lines[:2]]

        for i, line in enumerate(lines[2:]):
            raw = line.strip().split()
            raw[-1] = "~   wel_{0:02d}   ~".format(i)
            line = ' '.join(raw)
            f.write(line + '\n')

    df = pyemu.helpers.write_grid_tpl("gr_", os.path.join(new_d, "hk.ref.tpl"), "", shape=(m.nrow, m.ncol),
                                      spatial_reference=m.sr)
    os.chdir(new_d)
    pst = pyemu.Pst(base_pst)
    pst.parameter_data.drop(pst.par_names,inplace=True)
    pst.control_data.noptmax=0
    pst.template_files = []
    pst.input_files = []
    wel_df = pst.add_parameters("wel.dat.tpl", "wel.dat")
    pst.parameter_data.loc[wel_df.parnme,"partrans"] = "fixed" # for now...
    gr_df = pst.add_parameters("hk.ref.tpl", "hk.ref")
    pst.parameter_data.loc[gr_df.parnme,"parubnd"] = 1.0e+2
    pst.parameter_data.loc[gr_df.parnme, "parlbnd"] = 1.0e-2

    pst.write(new_pst)

    pst.write_input_files()
    os.chdir("..")
    with open(os.path.join(new_d, "model.bat")) as f:
        lines = f.readlines()
    with open(os.path.join(new_d, "model.bat"), 'w') as f:
        for line in lines:
            if "fac2real" in line or "hk.ref" in line:
                pass
            else:
                f.write(line)

    pyemu.os_utils.run("pestpp new.pst",cwd=new_d)

    gs = pyemu.geostats.read_struct_file(os.path.join(new_d, "struct.dat"))
    struct_dict = {}
    #struct_dict[gs[0]] = os.path.join(new_d, "hk.tpl")
    gs[1].variograms[0].contribution = 1.0
    struct_dict[gs[1]] = df
    cov = pyemu.helpers.geostatistical_prior_builder(pst, struct_dict=struct_dict,sigma_range=6.0)
    cov.to_binary(os.path.join(new_d, "prior.jcb"))
    pe = pyemu.helpers.geostatistical_draws(pst,struct_dict)
    pe.to_csv(os.path.join(new_d,"par.csv"))
    print(cov.shape)
    print(gs)

def add_extras_as_mults():

    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(base_d,new_d)

    m = flopy.modflow.Modflow.load("rect_model.nam",model_ws=new_d,verbose=True)

    with open(os.path.join(new_d,"wel.dat")) as f:
        lines = f.readlines()
    with open(os.path.join(new_d,'wel.dat.tpl'),'w') as f:
        f.write("ptf ~\n")
        [f.write(line) for line in lines[:2]]

        for i,line in enumerate(lines[2:]):
            raw = line.strip().split()
            raw[-1] = "~   wel_{0:02d}   ~".format(i)
            line = ' '.join(raw)
            f.write(line+'\n')

    df = pyemu.helpers.write_grid_tpl("gr_",os.path.join(new_d,"hk_grid.dat.tpl"),"",shape=(m.nrow,m.ncol),spatial_reference=m.sr)
    os.chdir(new_d)
    pst = pyemu.Pst(base_pst)
    pst.add_parameters("wel.dat.tpl","wel.dat")
    pst.add_parameters("hk_grid.dat.tpl","hk_grid.dat")
    pst.write(new_pst)
    pst.write_input_files()
    os.chdir("..")
    with open(os.path.join(new_d,"model.bat")) as f:
        lines = f.readlines()
    with open(os.path.join(new_d,"model.bat"),'w') as f:
        for line in lines:
            if "fac2real" in line:
                f.write(line)
                f.write("python mult.py\n")
            else:
                f.write(line)
    with open(os.path.join(new_d,"mult.py"),'w') as f:
        f.write("import numpy as np\n")
        f.write("mlt = np.loadtxt('hk_grid.dat')\n")
        f.write("vals = []\n")
        f.write("with open('hk.ref') as f:\n")
        f.write('    for line in f:\n')
        f.write("        vals.extend([float(r) for r in line.strip().split()])\n")
        f.write("base = np.array(vals).reshape(mlt.shape)\n")
        f.write("base *= mlt\n")
        f.write("np.savetxt('hk.ref',base,fmt='%15.6E')\n")

    os.chdir(new_d)
    sys.path.append('.')
    import mult
    os.system("model.bat")
    os.chdir('..')

    gs = pyemu.geostats.read_struct_file(os.path.join(new_d,"struct.dat"))
    struct_dict = {}
    struct_dict[gs[0]] = os.path.join(new_d,"hk.tpl")
    struct_dict[gs[1]] = df
    cov = pyemu.helpers.geostatistical_prior_builder(pst,struct_dict=struct_dict)
    cov.to_binary(os.path.join(new_d,"prior.jcb"))
    print(cov.shape)
    print(gs)

def base_ies():
    temp_d = "template_base_ies"
    if os.path.exists(temp_d):
        shutil.rmtree(temp_d)
    shutil.copytree(base_d,temp_d)
    os.remove(os.path.join(temp_d,"pestpp-ies.exe"))
    pst = pyemu.Pst(os.path.join(temp_d,base_pst))
    pst.pestpp_options = {}
    print(pst.control_data.noptmax)
    pst_name = "base_ies.pst"
    pst.write(os.path.join(temp_d,pst_name))
    pyemu.os_utils.start_slaves(temp_d,"pestpp-ies",pst_name,num_slaves=20,
                                master_dir=temp_d.replace("template","master"))

def run_ies(temp_d,pp_args = {},base_dd=new_d):
    if os.path.exists(temp_d):
        shutil.rmtree(temp_d)
    shutil.copytree(base_dd,temp_d)
    os.remove(os.path.join(temp_d,"pestpp-ies.exe"))
    try:
        pst = pyemu.Pst(os.path.join(temp_d,new_pst))
    except:
        pst = pyemu.Pst(os.path.join(temp_d, base_pst))
    pst.pestpp_options = {}
    # pst.pestpp_options["parcov"] = "prior.jcb"
    # pst.pestpp_options["ies_par_en"] = "par.csv"
    pst.pestpp_options["ies_bad_phi"] = 20000.0
    for k,v in pp_args.items():
        pst.pestpp_options[k] = v
    # pst.pestpp_options["parcov"] = "prior.jcb"
    # pst.pestpp_options["ies_par_en"] = "par.csv"
    pst.pestpp_options["ies_bad_phi"] = 20000.0

    pst.control_data.noptmax = 10
    pst_name = "new_ies.pst"
    pst.write(os.path.join(temp_d,pst_name))
    pyemu.os_utils.start_slaves(temp_d,"pestpp-ies",pst_name,num_slaves=20,
                                master_dir=temp_d.replace("template","master"))


def build_dist_localizer_grid(tol=None):
    sr = flopy.utils.SpatialReference.from_gridspec(os.path.join(new_d,"rect.spc"))
    pst = pyemu.Pst(os.path.join(new_d,new_pst))
    par = pst.parameter_data
    gr_par = par.loc[par.parnme.apply(lambda x :x.startswith("gr_")),:].copy()
    gr_par.loc[:, 'i'] = gr_par.parnme.apply(lambda x: int(x.split('_')[1][:3]))
    gr_par.loc[:, 'j'] = gr_par.parnme.apply(lambda x: int(x.split('_')[1][3:]))
    gr_par.loc[:, "x"] = gr_par.apply(lambda x: sr.xcentergrid[x.i,x.j],axis=1)
    gr_par.loc[:, "y"] = gr_par.apply(lambda x: sr.ycentergrid[x.i, x.j], axis=1)

    obs_df = pd.read_csv(os.path.join(new_d,"wells.crd"),delim_whitespace=True,header=None,names=["name","x","y","zone"])
    obs_df.index = obs_df.name
    v = pyemu.geostats.ExpVario(1.0,100.0)
    vecs = []
    for name in obs_df.name:
        vec = v.covariance_points(obs_df.loc[name,'x'],obs_df.loc[name,"y"],gr_par.x,gr_par.y)
        if tol is not None:
            vec[vec<tol] = 0.0
        vecs.append(vec)

    df = pd.DataFrame(vecs)
    df.index = obs_df.name + "_1"
    df.to_csv(os.path.join(new_d,"distance_localizer.csv"))

    # fig = plt.figure(figsize=(6,9))
    # ax = plt.subplot(111,aspect="equal")
    # arr = np.zeros((sr.nrow,sr.ncol))
    # arr /= arr.max()
    # arr[gr_par.i,gr_par.j] = df.sum()
    #
    # ax.pcolormesh(sr.xcentergrid,sr.ycentergrid,arr)
    # plt.show()


def plot_pdfs():
    master_ds = [d for d in os.listdir('.') if d.startswith("master_") and os.path.isdir(d)]
    posterior_dfs = {}
    for master_d in master_ds:
        files = [f for f in os.listdir(master_d) if f.endswith(".obs.csv") and not "base" in f]
        inum = [int(f.split('.')[1]) for f in files]
        post_file = {i:f for i,f in zip(inum,files)}[max(inum)]
        df = pd.read_csv(os.path.join(master_d,post_file),index_col=0)
        df.columns = df.columns.str.lower()
        posterior_dfs[master_d] = df
    pyemu.plot_utils.ensemble_helper(list(posterior_dfs.values()),plot_cols=forecasts,filename="ies_pdfs.pdf")


def run_all():
    #base ies case with pp
    run_ies("template_pp_ies", base_dd=base_d)

    #base ies grid case
    pp_args = {}
    pp_args["ies_par_en"] = "par.csv"
    run_ies("template_grid_ies", base_dd=new_d,pp_args=pp_args)

    #ies grid with distance localization - no cutoff
    build_dist_localizer_grid()
    pp_args["ies_localizer"] = "distance_localizer.csv"
    pp_args["ies_localize_how"] = "pars"
    run_ies("template_dist_loc_nocut_by_par",pp_args=pp_args)

    #same but by obs
    pp_args["ies_localize_how"] = "obs"
    run_ies("template_dist_loc_nocut_by_obs", pp_args=pp_args)

    # distance localization with cutoff
    build_dist_localizer_grid(tol=0.2)
    pp_args["ies_localize_how"] = "pars"
    run_ies("template_dist_loc_cut_by_par", pp_args=pp_args)

    pp_args["ies_localize_how"] = "obs"
    run_ies("template_dist_loc_cut_by_obs", pp_args=pp_args)


if __name__ == "__main__":
    #base_ies()
    #add_extras_without_pps()
    #grid_ies()
    #build_dist_localizer_grid()
    #run_ies("template_dist_loc_nocut",pp_args={"ies_par_en":"par.csv","ies_localizer":"distance_localizer.csv","ies_localize_how":"pars"})

    #run_ies("template_pp_ies",base_dd=base_d)
    run_all
    plot_pdfs()
