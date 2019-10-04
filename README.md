

<table>
    <tr>
        <th><img src="https://raw.githubusercontent.com/zbloss/whitebox/master/img/box.png" style='width: 75px; height: 75px'></th>
        <th><h1>WhiteBox</h1></th>
    </tr>
</table>

<h2>home for the project dedicated to providing insights into black box model explanation, interpretation, and degradation</h2>

<hr>

## Description
This is currently a work in progress project aimed at solving the need for model interpretation, explanation, and degradation.
I've noticed this to be a pain point in my professional work as well as many others. This project serves as a solution.

The goal is for this package to be model agnostic. Whether you're using Sci-kit Learn, Tensorflow, PyTorch, etc., I want
this pacakge to be utilized. 

### Current Plan
This project will be a dockerized web app that, given a model, the data it was trained on, and the data to be evaluated,
will provide users with 3 views:
1. Explaination: Here you will be able to check why a model made each prediction on a case-by-case level.
2. Interpretation: This view will serve as the home for model understanding. The idea is that you will be able to use this page
to understand how your model works and what features it is picking up on.
3. Degradation: Here you can track how well your model has performed over time. It will provide an interface for whether
you should consider retraining your model.

## Getting Started [Development]:
1. Clone this repository
2. Run `docker-compose build`
3. Run `docker-compose up`
   1. This will create the docker container and launch the jupyter notebook instance to develop on.


## Credits
Icon made by [Elias Bikbulatov]("https://www.flaticon.com/authors/elias-bikbulatov") from www.flaticon.com
