# Copyright (c) CryoRes Team, ZhangLab, Tsinghua University. All Rights Reserved

import io
import os


def vis_mask(mrc_path, level):
    dirname   = os.path.dirname(mrc_path)
    mapname   = os.path.basename(mrc_path)
    basename  = mapname.split('.')[0]
    suffix    = mrc_path.split(".")[-1]
    mask_path = f"{basename}_mask.{suffix}"
    out_path = f"{dirname}/{basename}_mask.cmd"
    
    cmd = "open " + mapname + "\n" + \
          "open " + mask_path + "\n" + \
          "vol #0 region all step 1 level " + str(level) + " style surface" + "\n" + \
          "vol #1 region all step 1 style surface" + "\n" + \
          "windowsize 800 650" + "\n" + \
          "scale 0.7" + "\n" + \
          "set bgColor white" + "\n" + \
          "volume #1 step 1 color 1,1,.7,.3" + "\n" + \
          "set bgTransparency" + "\n"

    with open(out_path,"w") as f:
        f.write(cmd)
    print("Chimera based commline file for mask visualizaiton: \n", out_path)
    

def vis_localres(mrc_path, min_ai, max_ai, level):
    dirname   = os.path.dirname(mrc_path)
    mapname   = os.path.basename(mrc_path)
    basename  = mapname.split('.')[0]
    suffix    = mrc_path.split(".")[-1]
    localres_path = f"{basename}_localres.{suffix}"
    out_path = f"{dirname}/{basename}_localres.cmd"
    
    cmd = "open " + mapname + "\n" + \
          "open " + localres_path + "\n" + \
          "vol #0 region all step 1 level "+ str(level) +" style surface" + "\n" + \
          "vol #1 region all step 1 style surface" + "\n" + \
          "~ modeldisp #1" + "\n" + \
          "windowsize 800 650" + "\n" + \
          "scale 0.7" + "\n" + \
          "set bgColor white" + "\n" + \
          "set bgTransparency" + "\n"
    if max_ai == min_ai:
        cmd = cmd + "scolor #0 volume #1 cmap " + str(min_ai) + ",red:100,white" + "\n" + \
              "colorkey  0.88,0.9  0.9,0.1 labelColor black 'bg' white '"+str(min_ai)+"' red" + "\n"
    else:
        deta_index = (max_ai - min_ai)/6
        orange_value = min_ai + deta_index
        yellow_value = orange_value + deta_index
        green_value = yellow_value + deta_index
        cyan_value = green_value + deta_index
        blue_value = cyan_value + deta_index

        cmd = cmd + "scolor #0 volume #1 cmap "+str(min_ai)+",red:"+str(orange_value)+",orange:"+ \
                str(yellow_value)+",yellow:"+str(green_value)+",green:"+str(cyan_value)+",cyan:"+ \
                str(blue_value)+",blue:"+str(max_ai)+",purple:100,white" + "\n" + \
                "colorkey  0.88,0.9  0.9,0.1 labelColor black 'bg' white '"+str(max_ai)+"' purple '" + \
                str(blue_value)+"' blue '"+str(cyan_value)+"' cyan '"+str(green_value)+"' green '" + \
                str(yellow_value)+"' yellow '"+str(orange_value)+"' orange '"+str(min_ai)+"' red" + "\n"
    
    with open(out_path,"w") as f:
        f.write(cmd)
    print("Chimera based commline file for local resolution visualizaiton: \n", out_path)
    
def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


