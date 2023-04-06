import torch
import torch.nn as nn
import numpy as np



class WaveformNet(nn.Module):
    def __init__(self,generator,output=128):
        super().__init__()

        self.audio_encoder = WaveformEncoder()
        if generator == "direct":
            self.decoder = DirectUpsampler(output)
        elif generator == "unet":
            self.decoder = UNet(output)
        else:
            raise Exception("Generator: use direct or unet")

    def forward(self, x1,x2,x3):
        
        x = self.audio_encoder(x1,x2,x3)
        x = self.decoder(x)
        
        return x
    
    
class encode_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, double=False):
        super(encode_block, self).__init__()
        self.double = double
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )
        if self.double: # Quick bugfix without testing. DDP will not work if parameters are defined which are not used in the computation.
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True)
                )

    def forward(self, x):
        x = self.down_conv(x)
        if self.double:
            x = self.conv(x)
        
        return x

class decode_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, double=False):
        super(decode_block, self).__init__()
        self.double = double
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding, stride = stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
            )
        
    def forward(self, x):
        x = self.up_conv(x)
        if self.double:
            x = self.conv(x)
        
        return x

class WaveformEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1  = encode_block(in_channels=3, out_channels=64,kernel_size=(1,228),padding=(0,114),stride=(1,2))            
        self.enc2  = encode_block(64, 64, (1,128), (0,64), (1,3),False)
        self.enc3  = encode_block(64, 128, (1,64), (0,32), (1,3),False)    
        self.enc4  = encode_block(128, 256, (1,32), (0,16), (1,3),False)
        self.enc5  = encode_block(256, 512, (1,16), (0,8), (1,3),False)
        self.enc6  = encode_block(512, 512, (1,8), (0,4), (1,3),False)
        self.enc7  = encode_block(512, 512, (1,4), (0,2), (1,3),False)
        self.enc8  = encode_block(512, 1024, (1,3), (0,1), (1,3),False)
        # output 1024x1x1

    def forward(self, x1,x2,x3):
        x0 = torch.cat([x1,x2,x3],1)
        
        x = self.enc1(x0)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        x = self.enc7(x)
        x = self.enc8(x)

        return x

class DirectUpsampler(nn.Module):
    def __init__(self,output=128):
        super().__init__()

        # 4x4
        self.dec1  = decode_block(1024,512,4,0,1,True)
        # 8x8
        self.dec2  = decode_block(512,512,4,1,2,True)
        # 16x16
        self.dec3  = decode_block(512,256,4,1,2,True)
        # 32x32
        self.dec4  = decode_block(256,128,4,1,2,True)
        # 64x64
        self.dec5  = decode_block(128,128,4 if output >= 64 else 3,
                                    1,2 if output >= 64 else 1,True)
        # 128x128
        self.dec6  = decode_block(128,64,4 if output == 128 else 3,
                                    1, 2 if output == 128 else 1,True)

        self.final = nn.Conv2d(64, 1, 1) 

    def forward(self, x):
        if len(x) != 4:
            x = x.view(-1,1024,1,1)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.dec6(x)
        x = self.final(x)
        
        return x



# class raw_to_depth(Dataset):
#     """Loads the Data sets for training/validation/testing"""

#     def __init__(self, samples=None, output=128):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
            
#         """
#         self.samples = samples
#         self.output = (output,output)


#     def __len__(self):
#         if self.samples:
#             return len(self.samples)
#         else:
#             return 1
          
#     def __getitem__(self, idx):
#         if self.samples == None:
#             size = 100
#             left_raw = np.random.uniform(size=size)
#             right_raw = np.random.uniform(size=size)
#             other_raw = np.random.uniform(size=size)
#             depth_raw = np.random.uniform(size=(128, 128))
#         else:
#             post_fix = '-arr.npy'
#             audio_file_path = path + audio_folder + '/' + self.samples[idx] + post_fix
#             depth_file_path = path + camera_depth_folder + '/'+ self.samples[idx] + '.npy'

#             audio_raw = np.load(audio_file_path)
#             left_raw = audio_raw[0][:2400]
#             right_raw = audio_raw[1][:2400]
#             other_raw = audio_raw[2][:2400]
#             depth_raw = np.load(depth_file_path)
        
#         depth_cutoff = DEPTH_CUTOFF #5 meter cutoff
#         depth_raw[depth_raw > depth_cutoff] = depth_cutoff
#         depth_raw = depth_raw / depth_cutoff

#         # depth_raw = depth_raw < 720
#         # depth_raw = depth_raw.astype('int')

#         depth = np.expand_dims(depth_raw,axis=0)
#         left = np.expand_dims(np.expand_dims(left_raw,axis=0),axis=0)
#         right = np.expand_dims(np.expand_dims(right_raw,axis=0),axis=0)
#         other = np.expand_dims(np.expand_dims(right_raw,axis=0),axis=0)
                
#         return torch.from_numpy(other).float(), torch.from_numpy(left).float(),torch.from_numpy(right).float(), torch.from_numpy(depth).float()