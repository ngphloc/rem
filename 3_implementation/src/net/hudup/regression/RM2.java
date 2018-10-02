package net.hudup.regression;

import java.util.List;

import net.hudup.regression.em.ui.graph.Graph;

/**
 * This interface represents an advanced regression model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface RM2 extends RM {

	
	/**
	 * Getting large statistics. Actually, this method calls {@link #getStatistics()}.
	 * @return large statistics.
	 */
	LargeStatistics getLargeStatistics();

	
	/**
	 * Executing by X statistics.
	 * @param xStatistic X statistics (regressors). The first element of this X statistics is 1.
	 * @return result of execution. Return null if execution is failed.
	 */
	double executeByXStatistic(double[] xStatistic);

	
	/**
	 * Extracting regressor (X).
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param index specified index. Index 0 is not included in the profile because this specified index is in internal indices.
	 * Index starts from 1. So index 0 always indicates to null. 
	 * @return regressor (X) extracted.
	 */
	VarWrapper extractRegressor(int index);

	
	/**
     * Getting list of regressors.
	 * In the most general case that each index is an mathematical expression, this method is focused.
     * @return list of regressors.
     */
    List<VarWrapper> extractRegressors();
    
    
    /**
     * Getting list of single regressors.
     * @return list of single regressors.
     */
    List<VarWrapper> extractSingleRegressors();

    
	/**
	 * Extracting value of regressor (X) from specified profile.
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param input specified input. It is often profile.
	 * @param index specified index. Index 0 is not included in the profile because this specified index is in internal indices.
	 * Index starts from 1. So index 0 always indicates to value 1. 
	 * @return value of regressor (X) extracted from specified profile.
	 */
	double extractRegressorValue(Object input, int index);

	
	/**
	 * Extracting statistical values of specified regressor.
	 * @param regressor specified regressor.
	 * @return statistical values of specified regressor.
	 */
	List<Double> extractRegressorStatistic(VarWrapper regressor);
	
	
	/**
	 * Extracting response variable (Z).
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @return response variable (Z) extracted.
	 */
	VarWrapper extractResponse();

	
	/**
	 * Transforming independent variable Z.
	 * In the most general case that each index is an mathematical expression, this method is not focused but is useful in some cases.
	 * @param z specified variable Z.
	 * @param inverse if true, there is an inverse transformation.
	 * @return transformed value of Z.
	 */
	Object transformResponse(Object z, boolean inverse);
	
	
	/**
	 * Creating 2D decomposed graph for regressor.
	 * @param regressor specified regressor.
	 * @return 2D decomposed graph.
	 */
    Graph createRegressorGraph(VarWrapper regressor);

    
    /**
     * Creating graph for response variable.
     * @return graph for response variable.
     */
    public Graph createResponseGraph();

    	
    /**
     * Creating error graph for response variable.
     * @return error graph for response variable.
     */
    Graph createErrorGraph();

    	
    /**
     * Creating graph related to response variable.
     * @return graphs related to response variable.
     */
    List<Graph> createResponseRalatedGraphs();

    
    /**
     * Calculating variance.
     * @return variance.
     */
    double calcVariance();
    
    
    /**
     * Getting correlation between real response and estimated response.
     * @return correlation between real response and estimated response.
     */
    public double calcR();
    
    
    /**
     * Calculating mean and variance of errors.
     * @return mean and variance of errors.
     */
    public double[] calcError();
    
    
}
