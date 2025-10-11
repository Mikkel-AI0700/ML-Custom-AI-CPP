# Scikit-Learn Recreated in C++

## 📘 Overview
This project aims to **recreate the core concepts of Scikit-Learn using C++**, with support from **Python** for dataset generation and **Bash** for automation.  
The goal is to understand how Scikit-Learn works under the hood while deepening C++ knowledge and exploring how modern machine learning concepts can be implemented in a low-level language.

## 🧠 Motivation
While Scikit-Learn is a high-level and well-optimized library built in Python (with C/C++ extensions), this project was created to:
- Learn **how Scikit-Learn’s internals**—such as algorithms, preprocessors, and model management—work.
- **Enhance C++ programming skills** by implementing complex ML concepts from scratch.
- Explore the challenges of building machine learning components without relying on high-level libraries.

## ⚙️ Project Structure
The project combines multiple languages for different responsibilities:
- **C++** → Core machine learning algorithms, preprocessors, and components.  
- **Python** → Dataset generation using Scikit-Learn’s dataset generators to ensure accurate and standardized data.  
- **Bash** → Automation scripts for dataset creation, file compilation, and experiment management.

### 🧩 Planned Components
- Core algorithms (e.g., Linear Regression, Logistic Regression, Decision Trees, etc.)
- Preprocessing utilities (e.g., scalers, encoders, imputers)
- Evaluation metrics (e.g., accuracy, MSE, R²)
- Hyperparameter tuning utilities
- Model persistence (save/load mechanisms)

## 🚧 Current Status
The project is **in its early stages** and under **active development**.  
There are **known and unknown bugs**, and several components are still experimental or incomplete.

Technical limitations are expected—C++ lacks some high-level libraries used in Python ML ecosystems, which introduces additional engineering challenges.

## 🧪 Usage

### Prerequisites
- C++17 or later  
- Python 3.x with Scikit-Learn installed  
- Bash shell environment  

### Basic Workflow
1. **Generate datasets** using Python scripts:
   ```bash
   bash generate_datasets.sh
