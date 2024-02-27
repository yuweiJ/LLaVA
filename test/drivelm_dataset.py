from nuscenes.nuscenes import NuScenes

def load_nuscenes_infos(root_path,
                        version='v1.0-train-qa'):
    nusc = NuScenes(version=version,
                    dataroot=root_path,
                    verbose=True)
    
    sample_rec = self.get('sample', sample_token)

if __name__=='__main__':
    root_path="/home/y/data/vlm/public_dataset/drivelm"
    load_nuscenes_infos(root_path)
