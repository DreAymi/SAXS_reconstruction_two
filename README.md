# SAXS_reconstruction_two
This project will introduce another method to do reconstruction with SAXS profile.
The main idea of this method is connecting two neural networks directly, an autoencoder and a iq2gene network.
Iq2gene network is used to transfer iq to gene. The decoder part of autoencoder is used to transfer gene to 3D structure. 


Environment dependence(Same in project SAXS_reconstruction)
1. python2
2. sastbx
3. tensorflow 1.14

How to run the project?
Do the following steps in sequence.
1. In directory "pisa_pdb", run download.py to download pisa pdb files.

2. Run changeaxis.py to rotate all pisa pdb structure to the same orientation. The results will be saved in directory "rotate_pisa_pdb".

3. Run function run_samermax_pdb() in generate_data.py. This step will resize all the rotated pisa pdb structure to a certain Rmax value(70, you can change it.). The results will be saved in directory "samermax_pisa_pdb".

4. Run function generate_aotoencoder_traindata() in generate_data.py. This step will generate autoencoder standard train set. The results "train.tfrecords" and "test.tfrecords" will be saved in directory "autoencoder_traindata".

5. Run auto_encoder.py in directory "train_autoencoder", to train autoencoder network. The well-trained model will be saved in directory "train_autoencoder/model"

6. Run function run_iq() in generate_data.py. This step will compute the iq profile of all the pisa pdb in directory "samermax_pisa_pdb". And save all the iq data in directory "pisa_data_iq"

7. Run function generate_gene() in generate_data.py. This step will get the latend vector of all the pisa pdb in well-trained autoencoder model. And save the results in directory "pisa_data_gene".

8. Run function generate_iq2gene_traindata() in generate_data.py. To generate the train data set of iq2gene network. It will be saved as "train_iq2gene.tfrecords" and "test_iq2gene.tfrecords" in directory "iq2gene_traindata".

9. Run train_iq_z.py in directory "train_iq2gene", to train iq2gene network. The well-trained model will be saved in directory "train_iq2gene/model".

10. Run main_iq2z.py in directory "test_run", to do reconstruction test.

What's more
You can run get_pr.py to compute the pr profile(instead of iq profile) of all the pisa pdb in directory "samermax_pisa_pdb". Then use pr data to train iq2gene network.



