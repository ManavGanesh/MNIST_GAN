# Architecture
class discriminator(nn.Module):
  def __init__(self, channels_img, features_d):
    super(discriminator, self).__init__()
    self.disc = nn.Sequential(
                                                                                  #image dimensions = N x channels_img x 48 x 48
        nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),  #o/p -> N x features_d x 24 x 24
        nn.LeakyReLU(0.2),
        self.block(features_d, features_d * 2, 4, 2, 1),                          #o/p -> N x features*2 x 12 x 12
        self.block(features_d * 2, features_d * 4, 4, 2, 1),                      #o/p -> N * features*4 x 6  x 6
        self.block(features_d * 4, features_d * 8, 4, 2, 1),                      #o/p -> N x features*8 x 3 x 3
        nn.Conv2d(features_d * 8, 1, kernel_size=3, stride=1, padding=0),         #o/p -> N x 1          x 1 x 1
        nn.Sigmoid()
      )
  def block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2),
        )
  def forward(self, x):
    return self.disc(x)

class generator(nn.Module):
  def __init__(self, z_dim, channels_img, features_g):
    super(generator, self).__init__()
    self.gen = nn.Sequential(

                                                                                                # i/p -> N x z_dim x 1 x 1
        self.block(z_dim, features_g * 16, 3, 1, 0),                                            #o/p -> N x features_g*16 x 3 x 3
        self.block(features_g * 16, features_g * 8, 4, 2, 1),                                   #o/p -> N x features_g*8 x 6 x 6
        self.block(features_g * 8, features_g * 4, 4, 2, 1),                                    #o/p -> N x features_g*4 x 12 x 12
        self.block(features_g * 4, features_g * 2, 4, 2, 1),                                    #o/p -> N x features_g*2 x 24 x 24
        nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),   #o/p -> N x channels_img x 48 x 48
        nn.Tanh(),
    )
  def block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
  def forward(self, x):
        return self.gen(x)
