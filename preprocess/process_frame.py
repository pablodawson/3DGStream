import os
import shutil
import logging

def check_database_valid(path):
    assert os.path.exists(path)
    assert os.path.exists(os.path.join(path, "images.bin"))
    assert os.path.exists(os.path.join(path, "cameras.bin"))
    assert os.path.exists(os.path.join(path, "points3D.bin"))
    assert os.path.exists(os.path.join(os.path.dirname(path), "input.db"))

def undistortFrame(output_path, offset, colmap_path="colmap", startframe=0):

    folder = os.path.join(output_path, "colmap_" + str(offset))
    assert os.path.exists(folder)

    source_folder = os.path.join(output_path, "colmap_" + str(startframe))

    inputimagefolder = os.path.join(folder, "input")

    source_distorted = os.path.join(source_folder, "distorted")
    source_sparse = os.path.join(source_folder, "sparse")

    destination_distorted = os.path.join(folder, "distorted")
    destination_sparse = os.path.join(folder, "sparse")

    # Copy both folders
    if not os.path.exists(destination_distorted):
        shutil.copytree(source_distorted, destination_distorted)
    if not os.path.exists(destination_sparse):
        shutil.copytree(source_sparse,destination_sparse)
        
    distortedmodel = os.path.join(folder, "distorted", "sparse")

    # Undistort input images
    img_undist_cmd = f"{colmap_path}" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + os.path.join(distortedmodel, "0") + " --output_path " + folder  \
    + " --output_type COLMAP" 
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    

def getColmapFrame(output_path, offset, colmap_path="colmap", manual=True, startframe=0, feature_matcher = "sift"):
    
    folder = os.path.join(output_path, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted", "sparse")

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    ## Feature extraction

    if feature_matcher == "sift":

        feat_extracton_cmd = colmap_path + f" feature_extractor --database_path {dbfile} --image_path {inputimagefolder} + \
            --ImageReader.camera_model " + "OPENCV" + " \
            --ImageReader.single_camera 1 \
            --SiftExtraction.use_gpu 1"
        exit_code = os.system(feat_extracton_cmd)
        if exit_code != 0:
            exit(exit_code)

        # Feature matching
        feat_matching_cmd = colmap_path + " exhaustive_matcher \
            --database_path " + dbfile + "\
            --SiftMatching.use_gpu 1"
        exit_code = os.system(feat_matching_cmd)
        if exit_code != 0:
            exit(exit_code)
    
    elif feature_matcher == "superpoint":
        cmd = "python deep-image-matching/main.py --images " + inputimagefolder + " --pipeline superpoint+lightglue" + \
        " --config deep-image-matching/config/superpoint+lightglue.yaml" + " --camera_options deep-image-matching/config/cameras.yaml --force"
        dbfile_sp = os.path.join(folder, "results_superpoint+lightglue_matching_lowres_quality_high", "database.db")
        distortedmodel_sp = os.path.join(folder, "results_superpoint+lightglue_matching_lowres_quality_high", "reconstruction")

        exit_code = os.system(cmd)
        if exit_code != 0:
            exit(exit_code)

        out_dir = os.path.join(distortedmodel, "0")
        os.makedirs(out_dir, exist_ok=True)

        shutil.copy(dbfile_sp, dbfile)
        for f in os.listdir(distortedmodel_sp):
            shutil.move(os.path.join(distortedmodel_sp, f), out_dir) # distortedmodel
    
    if manual:
        # Copy reconstruction from first frame
        source_folder = os.path.join(folder.replace(f"colmap_{offset}", f"colmap_{startframe}"), os.path.join("distorted","sparse","0"))
        if not os.path.exists(manualinputfolder):
            shutil.copytree(source_folder, manualinputfolder)
        
        check_database_valid(manualinputfolder)
        
        print("Starting triangulation")
        os.makedirs(os.path.join(distortedmodel, "0"), exist_ok=True)
        cmd = f"{colmap_path} point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + os.path.join(distortedmodel,"0" ) \
        + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001 --clear_points 1"
    else:
        cmd = colmap_path + " mapper \
        --database_path " + dbfile  + "\
        --image_path "  + inputimagefolder +"\
        --output_path "  + os.path.join(distortedmodel) + "\
        --Mapper.ba_global_function_tolerance=0.000001"
    
    exit_code = os.system(cmd)
    if exit_code != 0:
        logging.error(f"Failed with code {exit_code}. Exiting.")
        exit(exit_code)
    
    # sometimes it creates multiple reconstructions, we want to keep the largest one
    # no way this is necessary right?
    reconstructions = os.listdir(distortedmodel)
    if not manual and len(reconstructions) > 1:
        max_size = 0
        max_folder = None
        for r in reconstructions:
            size = sum(os.path.getsize(os.path.join(distortedmodel, r, f)) for f in os.listdir(os.path.join(distortedmodel, r)))
            if size > max_size:
                max_size = size
                max_folder = r
        source_folder = os.path.join(distortedmodel, max_folder)
        shutil.rmtree(os.path.join(distortedmodel, "0"))
        os.rename(source_folder, os.path.join(distortedmodel, "0"))
        
    # Undistort input images
    img_undist_cmd = f"{colmap_path}" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + os.path.join(distortedmodel, "0") + " --output_path " + folder  \
    + " --output_type COLMAP" 
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    
    files = os.listdir(os.path.join(folder, "sparse"))
    os.makedirs(os.path.join(folder, "sparse", "0"), exist_ok=True)
    for file in files:
        if file == "0":
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)
