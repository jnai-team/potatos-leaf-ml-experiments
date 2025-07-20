#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===============================================================================
#
# Copyright (c) 2025 <> All Rights Reserved
#
#
# File: /home/Administrator/projects/2025_03_01_zhangphd_paper/experiments/src/visual.py
# Author: Hai Liang Wang
# Date: 2025-03-19:19:00:28
#
# ===============================================================================

"""

"""
__copyright__ = "Copyright (c) 2020 . All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2025-03-19:19:00:28"

import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
from sklearn.metrics import accuracy_score, f1_score, recall_score
import plotly.graph_objects as go
import torch
from torchview import draw_graph


def go_figure(title, xaxis_title, yaxis_title, data, is_show=True,
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


def save_figure2jpg(fig, filepath, width=800, height=400):
    '''
    save figure file to image
    '''
    fig.write_image(file=filepath, format="jpg", width=width, height=height)


def export_onnx_archive(model, filepath, input_sample):
    '''
    Export model as onnx format file
    https://pytorch.org/docs/stable/onnx.html
    '''
    is_training = False

    if model.training:
        # view the network graph in onnx format
        # https://pytorch.org/docs/stable/onnx.html
        model.eval()
        is_training = True

    torch.onnx.export(
        model,        # model to export
        (input_sample,),      # inputs of the model,
        filepath,        # filename of the ONNX model
        input_names=["input"],  # Rename inputs for the ONNX model
        dynamo=True             # True or False to select the exporter to use
    )

    if not os.path.exists(filepath):
        raise BaseException("File %s not found" % filepath)

    if is_training:
        model.train()


def export_model_graph(
        model,
        input_sample,
        directory,
        filename="model_graph",
        format="svg",
        scale=5.0):
    '''
    Export model as image with torchview
    https://mert-kurttutan.github.io/torchview/reference/torchview/#torchview.torchview.draw_graph
    '''
    model_graph = draw_graph(model,
                             input_size=input_sample.shape,
                             expand_nested=True,
                             directory=directory)
    model_graph.visual_graph.node_attr["fontname"] = "Helvetica"
    model_graph.resize_graph(scale=scale)  # scale as per the view
    # https://graphviz.readthedocs.io/en/stable/api.html#graphviz.Graph
    model_graph.visual_graph.render(filename=filename, directory=directory, format=format)
    filepath = os.path.join(directory, "%s.%s" % (filename, format))

    if not os.path.exists(filepath):
        raise BaseException("Error on export torchview graph %s" % filepath)

    return filepath
