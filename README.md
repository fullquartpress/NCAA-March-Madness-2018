# Project 3 - Choose your own adventure

Due Date: March 14, 2018

As you progress through the course, the projects are going to give way to working on your capstone. In preparation for this transition, each instructor has chosen a dataset for you to simply play around with and explore.

From those three datasets, **you'll choose one** that is of interest to you. Use the skills you've learned up to this point (EDA, machine learning, etc), and see what you can come up with.

Unlike the previous projects, this one is intentionally left open-ended; there are no specific questions for you to answer and no set path that you have to follow. We're excited to see where your curiosity takes you.

Here are the three datasets:

## From David
A past hackathon hosted by Devpost, sponsored by AT&T, was conducted with a specific goal in mind:  Increase graduation rates of public high school studnets to %90 by the year 2020.  For the purpose of this analysis, you may assume how we may increase the graduation rate by any amount with no time restrictions (we're not assuming a timeseries anlaysis here basically).

_High level variables include_:

- Geographic charactistics
  - Income
  - Race
  - Family

There may be unknown factors to consider but in terms of what can be measured, you may want to start by measuring what can be observed about graduation rates overall.  Consider taking the approach of how you might apply resources given the characteristics you could measure through analysis (ie:  Focus on bringing people above the poverty line for instance over spending more money on teachers salaries -- for example).  Your job is to define what is possible to be measured, and recommend viable options, with some level of certainy that can be asserted through proper analysis.

### Data

For a description of the 550 variable features, check out this [data dictionary](https://www.census.gov/research/data/planning_database/2014/docs/PDB_Tract_2014-11-20a.pdf).  Each dataset, should come with its own data dictionary contained in each respective archived datasets. 

The data itself comes in 2 flavors, and you can choose either:

#### 1. Pre-built
Data that is created by the organizers, which is ready to go without any munging or joining.  At the lowest level, this dataset is a combined verion that includes a joined version by school districts from census tracts.  It can be modeled out of the box (after you've cleaned, etc).

#### 2. By Census region, not joined to school district
These datasets are much more granular but requires that you research and find some way to join / group subsets of it to a predefined list of school districts / singlular dataset.  While this is challenging, it is possible to come up with a dataset that can achieve a higher baseline R2 score for regression but also for other models.  For some intuition about how the pre-built dataset is created, check out the [benchmark documents provided on their website](https://challenges.s3.amazonaws.com/data_for_diplomas/Data%20for%20Diplomas%20Sample%20Benchmark%20Analyses.pdf).

> **Bonus** Can you find and include weather data to assert any impact on graduation.
> This could be scraped from the web or from an API if you choose.  It might be useful to consider regional patterns of weather given the timeframe of school schedules.  **Completely optional and no not a requirement to include.**

You may also research and look at participant solutions from the completed competition for this problem as a point of research on the Devpost website.

[Data For Diplomas](https://datafordiplomas.devpost.com/)

Full details about the datasets can be found here:

[DFD Resources Description](https://datafordiplomas.devpost.com/details/resources)

The actual datasets can be found here:
- [Pre-build dataset](https://challenges.s3.amazonaws.com/data_for_diplomas/Data%20for%20Diplomas_Merged%20Data.zip)
- [Separate datsets (needs to be merged)](https://challenges.s3.amazonaws.com/data_for_diplomas/Data%20for%20Diplomas_Unmerged%20Data.zip)

> **Tip** - You can create more than one model to describe different subsets and/or different aspects of your data, if you choose.  

## From Riley
If you're a fan of March Madness, this is the dataset for you! Every year, Kaggle hosts their "March Machine Learning Mania" competition, wherein you build a model to predict the winner of the NCAA men's basketball tournament. Last year, I finished in a respectable-yet-forgettable 57th place.

It's never too early to start building your model. The data you'll be using is current through last year's season, and will be updated when next year's competition starts (late Jan/early Feb?).

If you have a good workflow set up by then, all you'll need to do is download the latest CSVs and you'll be good to go!

> **WARNING**: When predicting the outcome of a game, you can use stats that occur up to **but not including** that particular game.

- https://www.kaggle.com/c/mens-machine-learning-competition-2018/data
- https://www.kaggle.com/c/womens-machine-learning-competition-2018/data

## Project Feedback + Evaluation

For all projects, students will be evaluated on a simple 4 point scale (0-3 inclusive). Instructors will use this rubric when scoring student performance on each of the core project requirements:

Score | Expectations
----- | ------------
**0** | _Does not meet expectations. Try again._
**1** | _Approaching expectations. Getting there..._
**2** | _Meets expecations. Great job._
**3** | _Surpasses expectations. Brilliant!_

For Project 3 the evaluation categories are as follows:

- **Organization**:	Clearly commented, annotated and sectioned Jupyter notebook or Python script. Comments and annotations add clarity, explanation and intent to the work. Notebook is well-structured with title, author and sections. Assumptions are stated and justified.
- **Exploratory Data Analysis**: A thorough exploratory data analysis has been conducted. Descriptive statistics, univariate and bivariate analysis, and plotting are skillfully used to explore connections across the dataset between features and targets.
- **Modeling Process**: Skillful and correct use of cross-validation, grid search, and goodness-of-fit metrics to evaluate candidate models. Assumptions and decisions in the modeling process are stated and justified. Use of correct modeling techniques in each challenge. Data is reproducibly and reliably transformed between training and test datasets.
