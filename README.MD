# Bachelor thesis - Implementing Job Reassignment Problem with Quantum Approximate Optimization Algorithm
### Universidad Nacional del Comahue, Argentina, Neuquén
### Project in progress

Student: Adriano Lusso

Director: Christian Nelson Gimenez

Codirector: Alejandro Mata Ali

## Description
This repository has the complete implementation (algorithms, processes and experiments) for the my Computer Science bachelor thesis. The mail goal was to implement the Job Reassignment Problem (JRP) into the Quantum Approximate Optimization Algorithm (QAOA).

While the thesis is still in progress, the structure of the alredy done work will be explained on the following sections.

## Introduction

In this folder, you will find the first and most basic notebook, for start getting into the project. Here, i describe the components of the JRP, made it first QUBO implementation and run some basic tests over the QAOA.

## Experimenting with the 'testQAOAsolver' workflow
$\textbf{The main experiment in the repository}$. Inside \real directory it can be find the real experiment, the results folder and the result notebook where the data postprocessing is been done. I am currently working on it, so it is uncompleted an may change during the following week.

## Functions

Here, they are defined the classes and functions used for further more complex experiments. Inside this folder, an 'old' folder will also be available. Here, there some older versions of some functions that had been used in some of the first experiments. While they aren't deleted so as to keep that experiments working, it is recommended to just use the python files in the 'functions' directory.

## OpenQAOA
 
This is just a duplicate of the OpenQAOA library repository. In some experiments, a local installation of the library will be done, so for that, the local repo is neccesary.

## OpenQAOA modified files
In the experiment with small instances, it has been discovered that a remote typical installation of OpenQAOA in Google Colab is not correctly done, due to dependencies prolem. This folder has some of the OpenQAOA files, but with the necessary modification needed for installing the tool on Google Colab.

## Eperiment with small instances

An auxiliar experiment where i try a few small JRP instances with Google Colab + GPU.


