import os
import sys
import shutil
import flopy
import pyemu

base_d = "template_org"
base_pst = "temp.pst"
new_d = "template_new"
new_pst = "new.pst"

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

def grid_ies():
    temp_d = "template_grid_ies"
    if os.path.exists(temp_d):
        shutil.rmtree(temp_d)
    shutil.copytree(new_d,temp_d)
    os.remove(os.path.join(temp_d,"pestpp-ies.exe"))
    pst = pyemu.Pst(os.path.join(temp_d,new_pst))
    pst.pestpp_options = {}
   # pst.pestpp_options["parcov"] = "prior.jcb"
    pst.pestpp_options["ies_par_en"] = "par.csv"
    pst.pestpp_options["ies_bad_phi"] = 2000.0
    pst.control_data.noptmax = 10
    pst_name = "new_ies.pst"
    pst.write(os.path.join(temp_d,pst_name))
    pyemu.os_utils.start_slaves(temp_d,"pestpp-ies",pst_name,num_slaves=20,
                                master_dir=temp_d.replace("template","master"))
if __name__ == "__main__":
    #base_ies()
    add_extras_without_pps()
    grid_ies()


