# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:28:56 2016

@author: jason
"""

data={'name':['Tim', 'Jim', 'Pam', 'Sam'],
      'age': [29, 31, 27, 35],
      'ZIP': ['02115','02130','67700','00100']
    }
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plt.pcolor(corr_flavors)
    plt.colorbar()
    plt.savefig("corr_flavors.pdf")
    
    corr_whisky = pd.DataFrame.corr(flavors.transpose())
    plt.figure(figsize=(10,10))
    plt.pcolor(corr_whisky)
    plt.axis("tight")
    plt.colorbar()
    plt.savefig("corr_whisky.pdf")
    
    whisky["Group"] = pd.Series(model.row_labels_, index=whisky.index)
    whisky = whisky.ix[np.argsort(model.row_labels_)]
    whisky=whisky.reset_index(drop=True)
    plt.figure(figsize=(14,7))
    plt.subplot(121)
    plt.pcolor(corr_whisky)
    plt.title("Original")
    plt.axis("tight")
    plt.subplot(122)
    plt.pcolor(correlations)
    plt.title("Rearranged")
    plt.axis("tight")
    plt.savefig("correlations.pdf")