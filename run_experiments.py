import os
import sys
import shutil
import flopy
import pyemu

base_d = "template_org"
base_pst = "temp.pst"

def add_extras():
    new_d = "template_new"
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
    pst = pyemu.Pst("temp.pst")
    pst.add_parameters("wel.dat.tpl","wel.dat")
    pst.add_parameters("hk_grid.dat.tpl","hk_grid.dat")
    pst.write("new.pst")
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


if __name__ == "__main__":
    #base_ies()
    add_extras()


