### Procedure to train a model for new function class

1. Fix error tolerance and pick adequate step sizes.
   - The function ``one_fun()`` in ``comparison_ode.py`` and using a ``PredictorConstODE`` might be helpful.
   - From my experience about 5-10 step sizes with linear or logarithmic spacing works well.
   - Alternatively, we can also start by fixing step sizes and then picking the error tolerance accordingly.
2. It might be helpful to use an input-scaler for certain function classes.
    - Fit the scaler using ``build_scaler()`` in ``scaling.py`` and save it as a file.
    - Load the fitted scaler before starting the training.
3. Execute the training of the model using the ``main()`` function in ``train_predictor_ode.py``.
The following tweaks may help to improve performance:
    - slowly reduce the learning rate 
    - play with the ``eps`` parameter to control exploration
    - use a memory
    - use a different rewards function (see ``reward_functions.py``)
    - different NN model parameters (loss function, optimizer, regularizer, network topology, ...)
    - ...
4. Evaluate the performance of the trained model.
    - The function ``one_fun()`` in ``comparison_ode.py`` can be used to provide an overview of the model's behavior.
    - For a precise performance evaluation use the function ``pareto_model()`` in ``comparison_ode.py``.
    - Compare the performance to RK45 by evaluating RK45 using ``pareto_ode45()`` and plot the comparison with ``plot_pareto()``.
5. Save the model as well as all the used parameters (error tolerance, step sizes, etc.) in a new directory.