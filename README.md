# Catalog Uplift Modeling – Project Workflow

[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/pyspark-ML-orange)](https://spark.apache.org/)
[![MLFlow](https://img.shields.io/badge/MLFlow-Tracking-success)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Data Collection](#2-data-collection)
3. [Environment Setup](#3-environment-setup)
4. [Data Loading](#4-data-loading)
5. [Exploratory Data Analysis](#5-simple-eda)
6. [Feature Engineering](#6-feature-engineering)
7. [Next Steps](#next-steps)

---

## 1. Problem Statement

Develop a data-driven targeting approach for catalog marketing that:

- Identifies **incremental purchasers** influenced by a catalog.
- Estimates the **size of incremental audience**.
- Produces a **ranked list** of customers by expected incremental lift.

---

## 2. Data Collection

Data sources:  
- Customer Master (DM)  
- Transaction-level sales data  

Data ingested from **SSMS** → **Lakehouse** via `kartheek_pipeline`.  

Schemas and column descriptions for DM & sales are detailed in the original workflow documentation.




---

## 3. Environment Setup

Install dependencies:
