# Astonomische Methoden (astronomical methods)

This project was created as part of the lecture 'Astronomische Methoden'.
The aim is to develop a simple neural network which predicts the redshift from the given bands.

The data were accessed using `astroquery.sdss`.

## Networks
Two different networks were trained. 
The first one determines the redshift of galaxies while the second one deals with quasars.

While the training worked sufficiently well for the galaxies, quasars showed stronger deviations.
Therefore, the network for the quasars was altered. This was done by simply adding a l2 regularization to the 
hidden layers. 

Since the influence in the galaxy network was neglectable it will be ignored for this one.

|          | normal | regularization |
|----------|--------|----------------|
| galaxies | ![valloss(redshift)](images/galaxies/normal/linreg_valloss(redshift).png)  |  ![valloss(redshift)](images/galaxies/regularization/linreg_valloss(redshift).png) |
| quasars  | ![valloss(redshift)](images/quasars/normal/linreg_valloss(redshift).png)  |  ![valloss(redshift)](images/quasars/regularization/linreg_valloss(redshift).png) |


## Loss
The relative squared error was chosen as loss. 
This should cause the relative uncertainty to stay constant over the redshift.
