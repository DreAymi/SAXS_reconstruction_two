# SAXS_reconstruction_two
This project will introduce another method to do reconstruction with SAXS profile.
The main idea of this method is connecting two neural networks directly, an autoencoder and a iq2gene network.
Iq2gene network is used to transfer iq to gene. The decoder part of autoencoder is used to transfer gene to 3D structure. 

Firstly, rotate all the pdb train structures into the same orientation. By doing this, all the


