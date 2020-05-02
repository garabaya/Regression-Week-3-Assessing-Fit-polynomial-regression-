# Regression Week 3: Assessing Fit (polynomial regression)

In this notebook you will compare different regression models in order to assess which model fits best. We will be using polynomial regression as a means to examine this topic. In particular you will:
* Write a function to take an SArray and a degree and return an SFrame where each column is the SArray to a polynomial value up to the total degree e.g. degree = 3 then column 1 is the SArray column 2 is the SArray squared and column 3 is the SArray cubed
* Use matplotlib to visualize polynomial regressions
* Use matplotlib to visualize the same polynomial degree on different subsets of the data
* Use a validation set to select a polynomial degree
* Assess the final fit using test data

We will continue to use the House data from previous notebooks.

# Fire up Turi Create


```python
import turicreate
```

Next we're going to write a polynomial function that takes an SArray and a maximal degree and returns an SFrame with columns containing the SArray to all the powers up to the maximal degree.

The easiest way to apply a power to an SArray is to use the .apply() and lambda x: functions. 
For example to take the example array and compute the third power we can do as follows: (note running this cell the first time may take longer than expected since it loads Turi Create)


```python
tmp = turicreate.SArray([1., 2., 3.])
tmp_cubed = tmp.apply(lambda x: x**3)
print(tmp)
print(tmp_cubed)
```

    [1.0, 2.0, 3.0]
    [1.0, 8.0, 27.0]


We can create an empty SFrame using turicreate.SFrame() and then add any columns to it with ex_sframe['column_name'] = value. For example we create an empty SFrame and make the column 'power_1' to be the first power of tmp (i.e. tmp itself).


```python
ex_sframe = turicreate.SFrame()
ex_sframe['power_1'] = tmp
print(ex_sframe)
```

    +---------+
    | power_1 |
    +---------+
    |   1.0   |
    |   2.0   |
    |   3.0   |
    +---------+
    [3 rows x 1 columns]
    


# Polynomial_sframe function

Using the hints above complete the following function to create an SFrame consisting of the powers of an SArray up to a specific degree:


```python
def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = turicreate.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1): 
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name] = poly_sframe['power_1'].apply(lambda x: x**power)
    return poly_sframe
```

To test your function consider the smaller tmp variable and what you would expect the outcome of the following call:


```python
print(polynomial_sframe(tmp, 3))
```

    +---------+---------+---------+
    | power_1 | power_2 | power_3 |
    +---------+---------+---------+
    |   1.0   |   1.0   |   1.0   |
    |   2.0   |   4.0   |   8.0   |
    |   3.0   |   9.0   |   27.0  |
    +---------+---------+---------+
    [3 rows x 3 columns]
    


# Visualizing polynomial regression

Let's use matplotlib to visualize what a polynomial regression looks like on some real data.


```python
sales = turicreate.SFrame('home_data.sframe/')
```

As in Week 3, we will use the sqft_living variable. For plotting purposes (connecting the dots), you'll need to sort by the values of sqft_living. For houses with identical square footage, we break the tie by their prices.


```python
sales = sales.sort(['sqft_living', 'price'])
```

Let's start with a degree 1 polynomial using 'sqft_living' (i.e. a line) to predict 'price' and plot what it looks like.


```python
poly1_data = polynomial_sframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price'] # add price to the data since it's the target
```

NOTE: for all the models in this notebook use validation_set = None to ensure that all results are consistent across users.


```python
model1 = turicreate.linear_regression.create(poly1_data, target = 'price', features = ['power_1'], validation_set = None)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 21613</pre>



<pre>Number of features          : 1</pre>



<pre>Number of unpacked features : 1</pre>



<pre>Number of coefficients    : 2</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 1.009580     | 4362074.696077     | 261440.790724                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



```python
#let's take a look at the weights before we plot
model1.coefficients
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">value</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">stderr</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">(intercept)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-43579.08525145019</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4402.689697427721</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">280.6227708858474</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.936398555132125</td>
    </tr>
</table>
[2 rows x 4 columns]<br/>
</div>




```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
         poly1_data['power_1'], model1.predict(poly1_data),'-')
```




    [<matplotlib.lines.Line2D at 0x7fe680c1d9e8>,
     <matplotlib.lines.Line2D at 0x7fe680c1db38>]




![png](output_24_1.png)


Let's unpack that plt.plot() command. The first pair of SArrays we passed are the 1st power of sqft and the actual price we then ask it to print these as dots '.'. The next pair we pass is the 1st power of sqft and the predicted values from the linear model. We ask these to be plotted as a line '-'. 

We can see, not surprisingly, that the predicted values all fall on a line, specifically the one with slope 280 and intercept -43579. What if we wanted to plot a second degree polynomial?


```python
poly2_data = polynomial_sframe(sales['sqft_living'], 2)
my_features = poly2_data.column_names() # get the name of the features
poly2_data['price'] = sales['price'] # add price to the data since it's the target
model2 = turicreate.linear_regression.create(poly2_data, target = 'price', features = my_features, validation_set = None)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 21613</pre>



<pre>Number of features          : 2</pre>



<pre>Number of unpacked features : 2</pre>



<pre>Number of coefficients    : 3</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.008588     | 5913020.984255     | 250948.368758                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



```python
model2.coefficients
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">value</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">stderr</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">(intercept)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">199222.4964446195</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7058.004835516299</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">67.99406406773976</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.287872013161773</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.038581231278915384</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0008982465470323439</td>
    </tr>
</table>
[3 rows x 4 columns]<br/>
</div>




```python
plt.plot(poly2_data['power_1'],poly2_data['price'],'.',
         poly2_data['power_1'], model2.predict(poly2_data),'-')
```




    [<matplotlib.lines.Line2D at 0x7fe6806f26a0>,
     <matplotlib.lines.Line2D at 0x7fe6806f2780>]




![png](output_28_1.png)


The resulting model looks like half a parabola. Try on your own to see what the cubic looks like:


```python
poly3_data = polynomial_sframe(sales['sqft_living'], 3)
my_features = poly3_data.column_names() # get the name of the features
poly3_data['price'] = sales['price'] # add price to the data since it's the target
model3 = turicreate.linear_regression.create(poly3_data, target = 'price', features = my_features, validation_set = None)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 21613</pre>



<pre>Number of features          : 3</pre>



<pre>Number of unpacked features : 3</pre>



<pre>Number of coefficients    : 4</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.015488     | 3261066.736008     | 249261.286346                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



```python
model3.coefficients
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">value</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">stderr</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">(intercept)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">336788.1179517966</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10661.015371317615</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-90.14762361186747</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10.622289184419227</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.08703671508097557</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.002966306231483158</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-3.839852119597755e-06</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.241749095900145e-07</td>
    </tr>
</table>
[4 rows x 4 columns]<br/>
</div>




```python
plt.plot(poly3_data['power_1'],poly3_data['price'],'.',
         poly3_data['power_1'], model3.predict(poly3_data),'-')
```




    [<matplotlib.lines.Line2D at 0x7fe6806e8ba8>,
     <matplotlib.lines.Line2D at 0x7fe6806e8c88>]




![png](output_32_1.png)


Now try a 15th degree polynomial:


```python
poly15_data = polynomial_sframe(sales['sqft_living'], 15)
my_features = poly15_data.column_names() # get the name of the features
poly15_data['price'] = sales['price'] # add price to the data since it's the target
model15 = turicreate.linear_regression.create(poly15_data, target = 'price', features = my_features, validation_set = None)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 21613</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.013692     | 2662308.584339     | 245690.511190                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



```python
model15.coefficients
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">value</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">stderr</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">(intercept)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">73619.75210522377</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">410.2874625479694</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.23045071443460427</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.588405424472302e-05</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-5.657018025607986e-09</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-4.570281308121097e-13</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.6636020659609474e-17</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.385847693136091e-21</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.1472310407255558e-25</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-4.6529358647462826e-30</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
</table>
[16 rows x 4 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>




```python
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
         poly15_data['power_1'], model15.predict(poly15_data),'-')
```




    [<matplotlib.lines.Line2D at 0x7fe68065de48>,
     <matplotlib.lines.Line2D at 0x7fe68065df28>]




![png](output_36_1.png)


**What do you think of the 15th degree polynomial? Do you think this is appropriate? If we were to change the data do you think you'd get pretty much the same curve? Let's take a look.**

# Changing the data and re-learning

We're going to split the sales data into four subsets of roughly equal size. Then you will estimate a 15th degree polynomial model on all four subsets of the data. Print the coefficients (you should use .print_rows(num_rows = 16) to view all of them) and plot the resulting fit (as we did above). The quiz will ask you some questions about these results.

To split the sales data into four subsets, we perform the following steps:
* First split sales into 2 subsets with `.random_split(0.5, seed=0)`. 
* Next split the resulting subsets into 2 more subsets each. Use `.random_split(0.5, seed=0)`.

We set `seed=0` in these steps so that different users get consistent results.
You should end up with 4 subsets (`set_1`, `set_2`, `set_3`, `set_4`) of approximately equal size. 


```python
(half_1, half_2) = sales.random_split(0.5, seed=0)
(set_1, set_2) = half_1.random_split(0.5, seed=0)
(set_3, set_4) = half_2.random_split(0.5, seed=0)
```

Fit a 15th degree polynomial on set_1, set_2, set_3, and set_4 using sqft_living to predict prices. Print the coefficients and make a plot of the resulting model.


```python
poly15_data = polynomial_sframe(set_1['sqft_living'], 15)
my_features = poly15_data.column_names() # get the name of the features
poly15_data['price'] = set_1['price'] # add price to the data since it's the target
model15 = turicreate.linear_regression.create(poly15_data, target = 'price', features = my_features, validation_set = None)
model15.coefficients.print_rows(num_rows=16, num_columns=4)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5404</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.014660     | 2195218.932305     | 248858.822200                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+-------------------------+-----------------------+
    |     name    | index |          value          |         stderr        |
    +-------------+-------+-------------------------+-----------------------+
    | (intercept) |  None |    223312.75024735887   |          nan          |
    |   power_1   |  None |    118.08612759009429   |          nan          |
    |   power_2   |  None |  -0.047348201136106494  |          nan          |
    |   power_3   |  None |  3.253103424735485e-05  |          nan          |
    |   power_4   |  None |  -3.323721525708726e-09 |          nan          |
    |   power_5   |  None |   -9.7583045756326e-14  |          nan          |
    |   power_6   |  None |  1.1544030339970148e-17 |          nan          |
    |   power_7   |  None |  1.0514586941310895e-21 | 9.834107444697464e-17 |
    |   power_8   |  None |  3.4604961652319043e-26 |          nan          |
    |   power_9   |  None | -1.0965445396033781e-30 |          nan          |
    |   power_10  |  None | -2.4203181214715775e-34 |          nan          |
    |   power_11  |  None | -1.9960120684331556e-38 | 7.530139806380401e-33 |
    |   power_12  |  None | -1.0770990387847978e-42 |          nan          |
    |   power_13  |  None |  -2.728628177229153e-47 |          nan          |
    |   power_14  |  None |  2.447826934581167e-51  |          nan          |
    |   power_15  |  None |  5.0197523270931016e-55 |          nan          |
    +-------------+-------+-------------------------+-----------------------+
    [16 rows x 4 columns]
    



```python
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
         poly15_data['power_1'], model15.predict(poly15_data),'-')
```




    [<matplotlib.lines.Line2D at 0x7fe6805d5358>,
     <matplotlib.lines.Line2D at 0x7fe6805d5438>]




![png](output_43_1.png)



```python
poly15_data = polynomial_sframe(set_2['sqft_living'], 15)
my_features = poly15_data.column_names() # get the name of the features
poly15_data['price'] = set_2['price'] # add price to the data since it's the target
model15 = turicreate.linear_regression.create(poly15_data, target = 'price', features = my_features, validation_set = None)
model15.coefficients.print_rows(num_rows=16, num_columns=4)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5398</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.020364     | 2069212.978547     | 234840.067186                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+-------------------------+----------------------+
    |     name    | index |          value          |        stderr        |
    +-------------+-------+-------------------------+----------------------+
    | (intercept) |  None |    89836.50773678801    |  1068153.0338730316  |
    |   power_1   |  None |    319.8069467541447    |  5161.370721704783   |
    |   power_2   |  None |   -0.10331539703285443  |  9.765764190591652   |
    |   power_3   |  None |  1.0668247603610658e-05 | 0.007986774844848023 |
    |   power_4   |  None |  5.7557709775688265e-09 |         nan          |
    |   power_5   |  None |  -2.54663464694922e-13  |         nan          |
    |   power_6   |  None | -1.0964134508136501e-16 |         nan          |
    |   power_7   |  None |  -6.364584415677271e-21 |         nan          |
    |   power_8   |  None |  5.5256041690461245e-25 |         nan          |
    |   power_9   |  None |  1.3508203898572387e-28 |         nan          |
    |   power_10  |  None |  1.1840818823275861e-32 |         nan          |
    |   power_11  |  None |  1.9834800062111657e-37 |         nan          |
    |   power_12  |  None |  -9.925335906219101e-41 |         nan          |
    |   power_13  |  None |  -1.608348470351073e-44 |         nan          |
    |   power_14  |  None |  -9.120060241709453e-49 |         nan          |
    |   power_15  |  None |  1.686366583204479e-52  |         nan          |
    +-------------+-------+-------------------------+----------------------+
    [16 rows x 4 columns]
    



```python
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
         poly15_data['power_1'], model15.predict(poly15_data),'-')
```




    [<matplotlib.lines.Line2D at 0x7fe68053de48>,
     <matplotlib.lines.Line2D at 0x7fe68053df28>]




![png](output_45_1.png)



```python
poly15_data = polynomial_sframe(set_3['sqft_living'], 15)
my_features = poly15_data.column_names() # get the name of the features
poly15_data['price'] = set_3['price'] # add price to the data since it's the target
model15 = turicreate.linear_regression.create(poly15_data, target = 'price', features = my_features, validation_set = None)
model15.coefficients.print_rows(num_rows=16, num_columns=4)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5409</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.022606     | 2269769.506523     | 251460.072754                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+-------------------------+------------------------+
    |     name    | index |          value          |         stderr         |
    +-------------+-------+-------------------------+------------------------+
    | (intercept) |  None |     87317.9795534432    |   1312239.9698394272   |
    |   power_1   |  None |    356.30491104641953   |   6366.570002421361    |
    |   power_2   |  None |   -0.16481744280817676  |   12.890135699293946   |
    |   power_3   |  None |  4.404249926875631e-05  |  0.014320866376643955  |
    |   power_4   |  None |   6.48234876349399e-10  | 9.596328147110607e-06  |
    |   power_5   |  None |  -6.752532265620608e-13 | 3.9261381225379545e-09 |
    |   power_6   |  None | -3.3684259273123967e-17 | 8.329476496006475e-13  |
    |   power_7   |  None |  3.609997042271464e-21  |          nan           |
    |   power_8   |  None |  6.469997256952375e-25  |          nan           |
    |   power_9   |  None |  4.2363938881230826e-29 |          nan           |
    |   power_10  |  None | -3.6214942566554345e-34 | 6.225887372764209e-28  |
    |   power_11  |  None |  -4.271195272909962e-37 | 8.913513084503276e-32  |
    |   power_12  |  None |  -5.614459718165853e-41 | 9.036816267021372e-36  |
    |   power_13  |  None |  -3.874527729174895e-45 | 7.859525708046308e-40  |
    |   power_14  |  None |   4.69430360106533e-50  | 3.867807896979562e-44  |
    |   power_15  |  None |  6.390458860118455e-53  | 7.755384726650742e-49  |
    +-------------+-------+-------------------------+------------------------+
    [16 rows x 4 columns]
    



```python
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
         poly15_data['power_1'], model15.predict(poly15_data),'-')
```




    [<matplotlib.lines.Line2D at 0x7fe680523780>,
     <matplotlib.lines.Line2D at 0x7fe680523860>]




![png](output_47_1.png)



```python
poly15_data = polynomial_sframe(set_4['sqft_living'], 15)
my_features = poly15_data.column_names() # get the name of the features
poly15_data['price'] = set_4['price'] # add price to the data since it's the target
model15 = turicreate.linear_regression.create(poly15_data, target = 'price', features = my_features, validation_set = None)
model15.coefficients.print_rows(num_rows=16, num_columns=4)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5402</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.020047     | 2314893.173826     | 244563.136754                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+-------------------------+------------------------+
    |     name    | index |          value          |         stderr         |
    +-------------+-------+-------------------------+------------------------+
    | (intercept) |  None |    259020.87944831268   |   1555657.2648813187   |
    |   power_1   |  None |    -31.72771619349126   |   10206.331532978036   |
    |   power_2   |  None |   0.10970276960581996   |   27.945340898358204   |
    |   power_3   |  None | -1.5838384727820846e-05 |  0.04233393833972426   |
    |   power_4   |  None |  -4.476606239124062e-09 | 3.961676931695559e-05  |
    |   power_5   |  None |  1.1397657348710987e-12 |  2.40253601675784e-08  |
    |   power_6   |  None |  1.9766912057280682e-16 | 9.444623626379352e-12  |
    |   power_7   |  None | -6.1578367894174755e-21 | 2.1424184602263764e-15 |
    |   power_8   |  None |  -4.880123041026591e-24 |          nan           |
    |   power_9   |  None |  -6.621867812721926e-28 |          nan           |
    |   power_10  |  None |  -2.706315833472915e-32 |          nan           |
    |   power_11  |  None |  6.723704116417694e-36  |          nan           |
    |   power_12  |  None |  1.7411564629955993e-39 |  9.88390219483489e-35  |
    |   power_13  |  None |  2.0918837568633785e-43 | 1.2401250941213685e-38 |
    |   power_14  |  None |  4.7801556582516095e-48 | 6.0982722216174955e-43 |
    |   power_15  |  None | -4.7453533306938734e-51 | 1.299544023952453e-47  |
    +-------------+-------+-------------------------+------------------------+
    [16 rows x 4 columns]
    



```python
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
         poly15_data['power_1'], model15.predict(poly15_data),'-')
```




    [<matplotlib.lines.Line2D at 0x7fe680496198>,
     <matplotlib.lines.Line2D at 0x7fe680496278>]




![png](output_49_1.png)


Some questions you will be asked on your quiz:

**Quiz Question: Is the sign (positive or negative) for power_15 the same in all four models?**
No

**Quiz Question: (True/False) the plotted fitted lines look the same in all four plots**
False

# Selecting a Polynomial Degree

Whenever we have a "magic" parameter like the degree of the polynomial there is one well-known way to select these parameters: validation set. (We will explore another approach in week 4).

We split the sales dataset 3-way into training set, test set, and validation set as follows:

* Split our sales data into 2 sets: `training_and_validation` and `testing`. Use `random_split(0.9, seed=1)`.
* Further split our training data into two sets: `training` and `validation`. Use `random_split(0.5, seed=1)`.

Again, we set `seed=1` to obtain consistent results for different users.


```python
(training_and_validation, testing) = sales.random_split(0.9, seed=1)
(training, validation) = training_and_validation.random_split(0.5, seed=1)
```

Next you should write a loop that does the following:
* For degree in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] (to get this in python type range(1, 15+1))
    * Build an SFrame of polynomial data of train_data['sqft_living'] at the current degree
    * hint: my_features = poly_data.column_names() gives you a list e.g. ['power_1', 'power_2', 'power_3'] which you might find useful for turicreate.linear_regression.create( features = my_features)
    * Add train_data['price'] to the polynomial SFrame
    * Learn a polynomial regression model to sqft vs price with that degree on TRAIN data
    * Compute the RSS on VALIDATION data (here you will want to use .predict()) for that degree and you will need to make a polynmial SFrame using validation data.
* Report which degree had the lowest RSS on validation data (remember python indexes from 0)

(Note you can turn off the print out of linear_regression.create() with verbose = False)


```python
def get_RSS(model, data, outcome):
    # First get the predictions
    predicted = model.predict(data);
    # Then compute the residuals/errors
    errors = outcome-predicted;
    # Then square and add them up    
    RSS = (errors*errors).sum();
    return(RSS)  
```


```python
from heapq import heappush, heappop
def lowest_RSS_degree_model (train_data_set, validation_data_set, feature, output_feature, degrees):
    if degrees>1 :
        RSSs = []
        models = []
        heap = []
        for degree in range (1, degrees+1):
            poly_data = polynomial_sframe(train_data_set[feature], degree)
            my_features = poly_data.column_names()
            poly_data[output_feature] = train_data_set[output_feature]
            model = turicreate.linear_regression.create(poly_data,
                                                        target = output_feature,
                                                        features = my_features,
                                                        validation_set = None,
                                                        verbose= False,
                                                        l2_penalty=0., 
                                                        l1_penalty=0.)
            RSS = get_RSS(model, polynomial_sframe(validation_data_set[feature], degree), validation_data_set[output_feature])
            #save RSS into a min heap
            heappush(heap, (RSS,degree))
            RSSs.append(RSS)
            models.append(model)
        min_RSS = min(RSSs)
        min_model = models[RSSs.index(min_RSS)]
        print(heap)
    return (min_model)
```

**Quiz Question: Which degree (1, 2, …, 15) had the lowest RSS on Validation data?**


```python
best_model = lowest_RSS_degree_model(training, validation, 'sqft_living', 'price', 15)
```

    [(592395859849004.5, 6), (592677914323034.9, 8), (598827152777892.9, 5), (598630662756922.8, 9), (609123922774459.5, 4), (616719668845846.8, 3), (605727492843186.8, 7), (676709739838073.2, 1), (607091004045995.0, 2), (5868658570329244.0, 10), (8.560309284673024e+16, 11), (2.1794004014702093e+17, 12), (3.259038639091958e+17, 13), (6.557335167248488e+17, 14), (7.322855169735261e+17, 15)]


**Now that you have chosen the degree of your polynomial using validation data, compute the RSS of this model on TEST data. Report the RSS on your quiz.**

**Quiz Question: what is the RSS on TEST data for the model with the degree selected from Validation data?**


```python
get_RSS(best_model, polynomial_sframe(testing['sqft_living'], 6), testing['price'])
```




    123989069495092.97




```python
best_model.coefficients
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">value</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">stderr</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">(intercept)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-138616.6434657832</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">62226.86451772267</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">942.5229675125695</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">124.71517714313305</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.7182838465772793</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.09107619749761804</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.000288023742327898</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.101699716301387e-05</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-5.220252694798691e-08</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.221555393589805e-09</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.3921964392805715e-12</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.1533770715441225e-13</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1.3597954959742855e-16</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.2268199412649673e-17</td>
    </tr>
</table>
[7 rows x 4 columns]<br/>
</div>


