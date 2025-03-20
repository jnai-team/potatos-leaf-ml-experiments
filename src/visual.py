#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2025 <> All Rights Reserved
#
#
# File: /home/Administrator/projects/2025_03_01_zhangphd_paper/experiments/src/visual.py
# Author: Hai Liang Wang
# Date: 2025-03-19:19:00:28
#
#===============================================================================

"""
   
"""
__copyright__ = "Copyright (c) 2020 . All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2025-03-19:19:00:28"

import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str


import plotly.graph_objects as go

# Get ENV
from env import ENV


def go_figure(title, xaxis_title, yaxis_title, data, is_show=True, \
            legend_x=0.05, legend_y=1.1, width=800, height=400):
    '''
    Make a figure with go
    '''
    # Create the figure for the chart
    fig = go.Figure()

    for x in data:
        fig.add_trace(go.Scatter(x=list(range(1, len(x["numbers"]) + 1)),
                                y=x["numbers"], mode=x["mode"], name=x["name"]))

    
    # Update the layout for better visualization
    fig.update_layout(title=title,
                      xaxis_title=xaxis_title,
                      yaxis_title=yaxis_title,
                      legend=dict(x=legend_x, y=legend_y),
                      width=width, height=height)

    if is_show:
        fig.show()

    return fig


def save_figure2img(fig, filepath, width=800, height=400):
    '''
    save figure file to image
    '''
    fig.write_image(file=filepath, format="jpg", width=width, height=height)