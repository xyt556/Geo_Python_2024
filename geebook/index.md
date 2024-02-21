---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Geospatial Data Science with Earth Engine and Geemap

[Google Earth Engine](https://earthengine.google.com) (GEE) is a cloud computing platform with a [multi-petabyte catalog](https://developers.google.com/earth-engine/datasets) of satellite imagery and geospatial datasets {cite}`Gorelick2017-mz`. During the past few years, GEE has become very popular in the geospatial community and it has empowered numerous environmental applications at local, regional, and global scales {cite}`Amani2020-vb,Boothroyd2020-fx,Tamiminia2020-df,Wu2019-at`. Since GEE became publicly available in 2010, there has been an exponential growth in the number of peer-reviewed journal publications empowered by GEE (see {numref}`ch00_gee_pubs`). Based on the most recent bibliometric analysis, there are 698 peer-reviewed journal publications with the word “Google Earth Engine” in the title and 1,779 publications with the word "Google Earth Engine" in either the title or abstract. In 2021, the number of publications with “Google Earth Engine” in the title or abstract reached almost 800, which is more than a 20-fold increase from the year 2014 with only 4 publications.

GEE provides both JavaScript and Python APIs for making computational requests to the Earth Engine servers. Compared with the comprehensive [documentation](https://developers.google.com/earth-engine) and interactive IDE (i.e., [GEE JavaScript Code Editor](https://code.earthengine.google.com)) of the GEE JavaScript API, the GEE Python API has relatively little documentation and limited functionality for visualizing results interactively. The **geemap** Python package was created to fill this gap {cite}`Wu2020-br`. It is built upon a number of open-source Python libraries, such as the [earthengine-api](https://pypi.org/project/earthengine-api/), [folium](https://python-visualization.github.io/folium/), [ipyleaflet](https://github.com/jupyter-widgets/ipyleaflet), and [ipywidgets](https://github.com/jupyter-widgets/ipywidgets). Geemap enables users to analyze and visualize Earth Engine datasets interactively within a Jupyter-based environment with minimal coding (see {numref}`ch00_geemap_gui`).

**Geemap** is intended for students and researchers, who would like to utilize the Python ecosystem of diverse libraries and tools to explore Google Earth Engine. It is also designed for existing GEE users who would like to transition from the GEE JavaScript API to Python API. Geemap provides an interactive graphical user interface for converting GEE JavaScripts to Python scripts without the need to write any code. It can save users a lot of time and effort by providing a simple interface for exploring and visualizing Earth Engine datasets.

```{figure} chapters/images/ch01_gee_pubs.jpg
---
name: ch00_gee_pubs
width: 825px
---
The number of journal publications empowered by Google Earth Engine.
```

```{figure} chapters/images/ch01_geemap_gui.jpg
---
name: ch00_geemap_gui
width: 825px
---
The geemap graphical user interface built upon ipyleaflet and ipywidgets.
```
