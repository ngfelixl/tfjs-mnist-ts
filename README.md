# TypeScript version of the Nodejs Mnist neural network

## Install dependencies

Requirements: Install NodeJS and Python on your machine. On Windows you will need either `Visual Studio` or run
`npm install -g windows-build-tools`. On Ubuntu install the package `build-essential`. Then run

```
npm install
```

## Usage

Build the distribution files first

```
npm run build
```

Start the script with

```
npm start
```

## Description

The script downloads the MNIST data.

It contains two different models called `dense` and `conv`. The dense model is a simple two dense layers model with a *relu* and a *softmax* activation. The `conv` model is more complex.

