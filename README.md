# Astonomische Methoden (astronomical methods)

THE NETWORK DOES AT THE MOMENT NOT WORK

This project was created as part of the lecture 'Astronomische Methoden'.
The aim is to develop a simple neural network which predicts the redshift from the given bands.

The data were accessed using `astroquery.sdss`.

## Networks
Two different networks were trained, mapping the magnitude of the optical bands to the redshift. 
The first one determines the redshift of galaxies while the second one deals with quasars.

While the training worked sufficiently well for the galaxies, quasars showed stronger deviations.
Therefore, the network for the quasars was altered. This was done by simply adding a l2 regularization to the 
hidden layers. 

Since the influence in the galaxy network was neglectable it will be ignored for this one.

## Loss
The relative squared error was chosen as loss. 
This should cause the relative uncertainty to stay constant over the redshift.


## Additional notes
The residues of the quasar network (with regularization) clearly shows a structure.
Since these are very small we expect this to be caused by very few data which are shifted away from the others. 
But this may need further discussion.

## Data
The data used is from the Sloan Digital Sky Survey (SDSS) DR16. To obtain randomly sampled redshifts the data
is sorted by right ascension before choosing the amount of data sets to download from the top of the list. 
The object class is resticted depending for which network the training data is used for. 
