package net.hudup.regression;

import java.util.List;

import net.hudup.regression.AbstractRM.VarWrapper;
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
	 * Extracting name of response variable (Z).
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @return text of response variable (Z) extracted.
	 */
	String extractResponseName();

	
	/**
	 * Executing by X statistics.
	 * @param xStatistic X statistics (regressors).
	 * @return result of execution. Return null if execution is failed.
	 */
	Object executeByXStatistic(double[] xStatistic);

	
	/**
	 * Creating 2D decomposed graph for regressor.
	 * @param xIndex X index. This index is started by 1.
	 * @return 2D decomposed graph.
	 */
    Graph createRegressorGraph(int xIndex);

    
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
     * Getting list of regressor expressions.
     * @return list of regressor expressions.
     */
    List<VarWrapper> getRegressorExpressions();
    
    
    /**
     * Getting list of regressors.
     * @return list of regressors.
     */
    List<VarWrapper> getRegressors();

    
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
