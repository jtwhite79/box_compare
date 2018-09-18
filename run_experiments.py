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

local_vario = pyemu.geostats.ExpVario(1.0,150.0) #same range as pp structure

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


def generate_grid_cov_and_ensemble():
    m = flopy.modflow.Modflow.load("rect_model.nam", model_ws=new_d, verbose=True)
    pst = pyemu.Pst(os.path.join(new_d,new_pst))
    par = pst.parameter_data
    gr_par = par.loc[par.parnme.apply(lambda x: x.startswith("gr_")),:].copy()
    gr_par.loc[:,"i"] = gr_par.parnme.apply(lambda x : int(x.split('_')[1][:3]))
    gr_par.loc[:, "j"] = gr_par.parnme.apply(lambda x: int(x.split('_')[1][3:6]))
    gr_par.loc[:,'x'] = gr_par.apply(lambda x : m.sr.xcentergrid[x.i,x.j], axis=1)
    gr_par.loc[:, 'y'] = gr_par.apply(lambda x: m.sr.ycentergrid[x.i, x.j], axis=1)


    gs = pyemu.geostats.read_struct_file(os.path.join(new_d, "struct.dat"))
    struct_dict = {}
    #struct_dict[gs[0]] = os.path.join(new_d, "hk.tpl")
    gs[1].variograms[0].contribution = 1.0
    struct_dict[gs[1]] = gr_par
    cov = gs[1].covariance_matrix(gr_par.x,gr_par.y,gr_par.parnme)
    #cov = pyemu.helpers.geostatistical_prior_builder(pst, struct_dict=struct_dict,sigma_range=6.0)
    cov.to_binary(os.path.join(new_d, "prior.jcb"))
    #pe = pyemu.helpers.geostatistical_draws(pst,struct_dict)
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst,cov,num_reals=200,enforce_bounds=True)
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

def run(temp_d,pp_args = {},base_dd=new_d,exe="pestpp-ies",noptmax=10):
    if os.path.exists(temp_d):
        shutil.rmtree(temp_d)
    shutil.copytree(base_dd,temp_d)
    os.remove(os.path.join(temp_d,"pestpp-ies.exe"))
    try:
        pst = pyemu.Pst(os.path.join(temp_d,new_pst))
        pst_name = new_pst
    except:
        pst = pyemu.Pst(os.path.join(temp_d, base_pst))
        pst_name = base_pst
    pst.pestpp_options = {}
    # pst.pestpp_options["parcov"] = "prior.jcb"
    # pst.pestpp_options["ies_par_en"] = "par.csv"

    for k,v in pp_args.items():
        pst.pestpp_options[k] = v
    # pst.pestpp_options["parcov"] = "prior.jcb"
    # pst.pestpp_options["ies_par_en"] = "par.csv"
    #pst.pestpp_options["ies_bad_phi"] = 20000.0

    pst.control_data.noptmax = noptmax
    #pst.observation_data.loc[pst.nnz_obs_names,"weight"] = 30.0
    #pst_name = "new_ies.pst"
    pst.write(os.path.join(temp_d,pst_name))
    pyemu.os_utils.start_slaves(temp_d,exe,pst_name,num_slaves=20,
                                master_dir=temp_d.replace("template","master"))


def build_phy_localizer_grid(tol=None):
    m = flopy.modflow.Modflow.load("rect_model.nam", model_ws=new_d, verbose=True)
    # first run noptmax=-1 for pp model
    temp_d = "template_pp_jco"
    if not os.path.exists(temp_d):
        run("template_pp_jco",base_dd=base_d,exe="pestpp",noptmax=-1)
    master_d = temp_d.replace("template","master")
    pst_pp = pyemu.Pst(os.path.join(master_d,base_pst))
    pst_grid = pyemu.Pst(os.path.join(new_d,new_pst))

    # raise to the 10 power for the log transform
    jco_pp = pyemu.Jco.from_binary(os.path.join(master_d,base_pst.replace(".pst",".jcb"))).to_dataframe()
    pp_df = pyemu.pp_utils.pp_file_to_dataframe(os.path.join(master_d,"hk.pts"))
    #pp_df.index = pp_df.name
    #pp_df.loc[:,"name"] = "k_" + pp_df.name
    #pp_df.index = pp_df.name
    col_names = jco_pp.columns.copy()
    jco_pp.columns = jco_pp.columns.map(lambda x: x.replace("k_",""))
    pnames = []
    for i in range(m.nrow):
        for j in range(m.ncol):
            pnames.append("gr_{0:03d}{1:03d}".format(i, j))
    dfs = []
    for oname in pst_pp.nnz_obs_names:
        pp_df.loc[:,'parval1'] = jco_pp.loc[oname,pp_df.name].values
        pp_df.loc[pp_df.parval1 <1.0e-10,"parval1"] = 1.0e-10
        pp_df.fillna(1.0e-10)
        arr = pyemu.geostats.fac2real(pp_df,os.path.join(master_d,"factors.dat"),out_file=None)
        plt.imshow(arr)
        plt.show()
        df = pd.DataFrame(arr.flatten(),index=pnames,columns=[oname])
        dfs.append(df)

    df = pd.concat(dfs,axis=1)
    print(df)
    df.T.to_csv(os.path.join(new_d,"phy_localizer.csv"))






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

    vecs = []
    for name in obs_df.name:
        vec = local_vario.covariance_points(obs_df.loc[name,'x'],obs_df.loc[name,"y"],gr_par.x,gr_par.y)
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
    phi_dfs = {}
    for master_d in master_ds:
        #if not  "pp" in master_d:
        #    continue
        files = [f for f in os.listdir(master_d) if f.endswith(".obs.csv") and not "base" in f and not "mean" in f]
        try:
            inum = [int(f.split('.')[1]) for f in files]
        except:
            print(master_d,files)
            continue
        try:
            post_file = {i:f for i,f in zip(inum,files)}[max(inum)]
        except:
            continue
        df = pd.read_csv(os.path.join(master_d,post_file),index_col=0)
        df.columns = df.columns.str.lower()
        posterior_dfs[master_d] = df
        files = [f for f in os.listdir(master_d) if f.endswith(".phi.actual.csv")]

        df = pd.read_csv(os.path.join(master_d,files[0]))
        phi_dfs[master_d] = df
    fig = plt.figure(figsize=(10,10))
    ax_phi = plt.subplot(211)
    ax = plt.subplot(212)


    fname = "part_time"
    for d,df in posterior_dfs.items():
        lab = d.replace("master_","") + ", std:{0:6.1f}".format(df.loc[:,fname].std())
        df.loc[:,fname].hist(ax=ax,bins=20,normed=True,alpha=0.5,label=lab)
        df = phi_dfs[d]
        df.iloc[-1,6:].hist(ax=ax_phi,bins=20,normed=True,alpha=0.5,label=d)
    #pyemu.plot_utils.ensemble_helper(list(posterior_dfs.values()),plot_cols=forecasts,filename="ies_pdfs.pdf",)
    ylim = ax.get_ylim()
    ax.plot([3256,3256],ylim,"k--",lw=3.0)
    ax.set_xlabel("part time")
    ax_phi.set_xlabel("phi")
    ax_phi.set_xlim(0,100)
    ax.set_yticklabels([])
    ax_phi.set_yticklabels([])

    plt.legend()
    plt.show()

def run_all():

    #run("template_pp_base",base_dd=base_d,exe="pestpp")


    # base ies case with pp
    pp_args = {"ies_num_reals": 50}
    pp_args["ies_save_lambda_en"] = True
    #pp_args["ies_par_en"] = "random.csv"
    pp_args["parcov"] = "cov.mat"
    run("template_pp_ies", base_dd=base_d, pp_args=pp_args)

    pp_args["ies_use_prior_scaling"] = True
    pp_args["ies_use_approx"] = False
    run("template_pp_ies_full_scale", base_dd=base_d,pp_args=pp_args)

    #base ies grid case
    pp_args["parcov"] = "prior.jcb"
    pp_args["ies_bad_phi"] = 20000.0
    pp_args["ies_par_en"] = "par.csv"
    run("template_grid_ies", base_dd=new_d,pp_args=pp_args)

    # physics based localization
    build_phy_localizer_grid()
    pp_args["ies_localize_how"] = "pars"
    pp_args["ies_localizer"] = "phy_localizer.csv"
    run("template_phy_loc_by_par", pp_args=pp_args)

    # physics based localization with prior scaling and full solution
    pp_args["ies_use_prior_scaling"] = True
    pp_args["ies_use_approx"] = False
    run("template_phy_loc_by_par_full_scale", pp_args=pp_args)
    return

    #ies grid with distance localization - no cutoff
    build_dist_localizer_grid()
    pp_args["ies_localizer"] = "distance_localizer.csv"
    pp_args["ies_localize_how"] = "pars"
    run("template_dist_loc_nocut_by_par",pp_args=pp_args)

    #same but by obs
    pp_args["ies_localize_how"] = "obs"
    run("template_dist_loc_nocut_by_obs", pp_args=pp_args)

    # distance localization with cutoff
    build_dist_localizer_grid(tol=0.2)
    pp_args["ies_localize_how"] = "pars"
    run("template_dist_loc_cut_by_par", pp_args=pp_args)

    pp_args["ies_localize_how"] = "obs"
    run("template_dist_loc_cut_by_obs", pp_args=pp_args)

    pp_args["ies_localize_how"] = "obs"
    run("template_phy_loc_by_obs", pp_args=pp_args)


def start():
    pyemu.os_utils.start_slaves(new_d,"pestpp-ies","new.pst",num_slaves=20,slave_root=".",port=4030)


def invest():
    temp_d = "test"
    base_dd = new_d
    if os.path.exists(temp_d):
        shutil.rmtree(temp_d)
    shutil.copytree(base_dd,temp_d)
    os.remove(os.path.join(temp_d,"pestpp-ies.exe"))

    pst = pyemu.Pst(os.path.join(temp_d,new_pst))
    pst_name = new_pst
    pst.pestpp_options["ies_par_en"] = "par.csv"
    pst.control_data.noptmax = 1
    #pst.observation_data.loc[pst.nnz_obs_names,"weight"] = 30.0
    pst_name = "new_ies.pst"

    pst.write(os.path.join(temp_d,pst_name))
    #pyemu.os_utils.start_slaves(temp_d,exe,pst_name,num_slaves=20,
    #                            master_dir=temp_d.replace("template","master"))
    pyemu.os_utils.run("pestpp-ies {0}".format(pst_name),cwd=temp_d)
if __name__ == "__main__":
    #add_extras_without_pps()
    #build_dist_localizer_grid()
    build_phy_localizer_grid()
    #generate_grid_cov_and_ensemble()
    #run_all()
    #plot_pdfs()
    #start()
    #invest()