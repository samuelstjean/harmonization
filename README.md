# Harmonization of diffusion MRI datasets with adaptive dictionary learning

For details on the algorithm, you can check out the [published version](https://onlinelibrary.wiley.com/doi/10.1002/hbm.25117)

The manuscript detailing the original harmonization challenge which lead to this algorithm is also available from the publisher [here](https://www.sciencedirect.com/science/article/pii/S1053811919300837).

## How to install

To install a precompiled version, simply run `pip install dmri-harmonization`.

There is also a Dockerfile which will compile the code for you internally, see https://docs.docker.com/get-started/ for more details.

Normally everything is available pre-compiled, but feel free to redo it from source on your computer cluster for example.


## A step by step example

There are two main scripts to use, one to build the database and one to harmonize the datasets afterwards.

You will most likely want you data to be organized sensibly and have the same filename for every diffusion dataset under various subjects folders, just like in the example that can be found  under the tests folder.

This looks like this and is the format used by the BIDS standard and the HCP datasets. You do not need to follow this exactly, you just need to ensure that the filenames are somewhat consistent since everything is found internally by pattern matching and substitutions.

~~~bash
datasets/
├── subj1
│   ├── dwi_brain_mask.nii.gz
│   ├── dwi.bvals
│   ├── dwi.bvecs
│   └── dwi.nii.gz
├── subj2
│   ├── dwi_brain_mask.nii.gz
│   ├── dwi.bvals
│   ├── dwi.bvecs
│   └── dwi.nii.gz
~~~

~~~bash
datasets_bids/
├── subj1-bids
│   └── dwi
│       ├── subj1-dwi_brain_mask.nii.gz
│       ├── subj1-dwi.bvals
│       ├── subj1-dwi.bvecs
│       └── subj1-dwi.nii.gz
└── subj2-bids
    └── dwi
        ├── subj2-dwi_brain_mask.nii.gz
        ├── subj2-dwi.bvals
        ├── subj2-dwi.bvecs
        └── subj2-dwi.nii.gz
~~~

### Step 1
First write a config file like this

~~~bash
harmonization_get_global_D write myconfig.yaml
~~~

Open up this `myconfig.yaml` file and change a couple of options, namely the paths at the top to point to your datasets folder.

You can look up the example I mentioned above for a working example, but this should get you going.
There are a couple more options to play with, but the default should be sane.
I'd recommend looking at the numbers of cores option at the end however if you share a compute server with other people.

### Step 2
Once you have your config file set up, run the command (note how there is no `write` keyword anymore)

~~~bash
harmonization_get_global_D myconfig.yaml
~~~

Everything will be read from `myconfig.yaml`, including the input and output folder.
You'll see that all your datasets will be loaded in before processing, so you can double check the paths were set up correctly (particularly the glob option if you use BIDS with different filenames).

### Step 3
After running step 2, you now have an output dictionary file. Now run the command

~~~bash
harmonization_harmonize_my_data myconfig.yaml
~~~

Sit back and relax, and you should have your harmonized datasets in the folder you specified.
Remember you can also specify the same input and output folder to have the data side by side.

There are a few safeguards to not shoot yourself in the foot, like not overwriting datasets by default.
On subsequent runs, be sure to change the output dictionary filename as it sets the logic for the output filenames or set the overwrite option to `True` in the config file if desired.
