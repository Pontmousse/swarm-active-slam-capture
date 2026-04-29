import pybullet as p
import pybullet_data

# Start PyBullet in graphical mode (or DIRECT for non-graphical mode)
p.connect(p.GUI)

# Optional: Set the path to PyBullet data
p.setAdditionalSearchPath(pybullet_data.getDataPath())


coarse = True
object_name = 'Separation_Stage'

# Path to your STL file
input_obj = 'Targets/'+object_name+'/'+object_name+'.obj'

# Perform convex decomposition using V-HACD
if coarse == True:
    # Output file name for the convex shapes
    output_obj = 'Targets/'+object_name+'/'+object_name+'_vhacd_coarse.obj'

    # Log file
    log_file = 'Targets/'+object_name+'/'+object_name+'_vhacd_coarse_log.txt'

    # Coarse Decomposition
    p.vhacd(input_obj, output_obj, log_file, alpha=0.3, resolution=10000)

else:
    # Output file name for the convex shapes
    output_obj = 'Targets/'+object_name+'/'+object_name+'_vhacd_fine.obj'

    # Log file
    log_file = 'Targets/'+object_name+'/'+object_name+'_vhacd_fine_log.txt'

    # Fine Decomposition
    p.vhacd(input_obj, output_obj, log_file, 
        alpha=0.04, beta=0.005, gamma=0.0005, resolution=1000000, 
        concavity=0.0001, planeDownsampling=1, convexhullDownsampling=1, 
        convexhullApproximation=1, maxNumVerticesPerCH=256, pca=0, mode=0)


# Disconnect PyBullet
p.disconnect()