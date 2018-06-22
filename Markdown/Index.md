
# Jupyter Site

JupyterSite allows you to easily build a suite of rendered documents from Jupyter
notebooks. Jupyter notebooks go in, html/markdown pages, PDFs, slideshows, and tex
comes out. If you use this in a Github repository, it will even build a Github pages
site. This also shows how to make the notebooks easily accessible in Julia.

It just takes one button, simplifying the entire process. The `publish.sh` file
does all the heavy lifting, showing you how to do all of this (any of the conversions
can be disabled by commenting out the appropriate lines).

## Example Site

An example of the product this package generates can be found at: https://github.com/UCIDataScienceInitiative/IntroToJulia . 

The site that this generates is: http://UCIDataScienceInitiative.github.io/IntroToJulia/

## Installation

To use this, first install Jupyter (note: Python must be installed)

```bash
pip install jupyter
```

For the PDF output to work, a distribution of LaTeX must be installed. Also, we
will need pandoc:

```bash
pip install pandoc
```

## Using the Package

### Downloading the Template

The easiest way to get started is to clone the package repository using Git:

```bash
git clone https://github.com/ChrisRackauckas/JupyterSite
```

### The Notebooks

The files which the site is built from are the notebooks that are included in
the Notebooks folder. The `Index.ipynb` file build the site index (the first
page shown in the webpage) and additional `.ipynb` files are rendered in the various
forms. Thus to add your own content, simply add the `.ipynb` files to this directory.

Note that notebooks have to be setup for slides in order for the slideshow to work.
Inside the notebook, use view > CellToolbar > Slideshow and set the appropriate blocks
to slides.

### Building The Site

To build the site files, use the `publish.sh` file. To do so, go to the top directory
of the repository and use the command:

```bash
sh publish.sh
```

If your Git is correctly setup, this will render the files and upload the files to
Github. Upon success, your `index.html` will be available at:

```
http://<github-username>.github.io/<repository-name>/
```

For example, my user name is ChrisRackauckas and this repository is named JupyterSite.
Therefore the site for that this makes is at

```
http://ChrisRackauckas.github.io/JupyterSite/
```

For organizations, the user name is replaced with the organization name.

#### Caveat

Note that PDF output is not compatible with the usage of Markdown images (though
code with images will allow the PDFs to build, but the images will not appear).

## HTML
https://terenceliu98.github.io/JupyterSite//Html/Use_PY_in_Linear_Algebra.html
<br>

https://terenceliu98.github.io/JupyterSite//Html/Use_PY_in_Calulus.html
<br>

https://terenceliu98.github.io/JupyterSite//Html/Use_PY_in_Advanced_Statstics.html
## Markdown

https://terenceliu98.github.io/JupyterSite/tree/master/Markdown

## LaTeX

https://terenceliu98.github.io/JupyterSite/tree/master/Tex

## PDF

https://terenceliu98.github.io/JupyterSite/tree/master/Pdfs
