/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression;

import java.rmi.RemoteException;
import java.util.List;

import net.hudup.core.alg.ExecutableAlgRemoteTask;
import net.hudup.core.logistic.xURI;
import net.rem.regression.em.ui.graph.Graph;

/**
 * This interface declares methods for remote regression algorithm.
 * 
 * @author Loc Nguyen
 * @version 2.0
 *
 */
public interface RMRemoteTask extends ExecutableAlgRemoteTask {


	
	/**
	 * Name of regression indices field.
	 */
	final static String R_INDICES_FIELD = "r_indices";

	
	/**
	 * Default regression indices field. Each index can be number, field name, or mathematical expression. If index is number, it starts with 1 because number index 0 always points to 1 value.
	 * Number index -1 points to nothing.
	 * The pattern is {1, 2}, {#x3, 4, 5}, {x5, 6}, {log(x5), 6, 7, 8}, {#x9^#x10, 10}.
	 * The pattern can also be 1, 2, #x3+#x4, 5, x6, 7, 8, log(x9), #y^2.
	 * However, it is impossible to specify 1^2, log(1) but it is possible to specify log(x1), x2.
	 */
	final static String R_INDICES_DEFAULT = "{1, #x2, -1, (#x3 + #x4)^2, log(#y)}"; //Use default indices in which n-1 first variables are regressors and the last variable is response variable
	
	
	/**
	 * Guidance note.
	 */
	final static String note = "The attribute \"r_indices\" indicates indices of independent variables (regressors) and dependent variables (responsors).\n" +
			"Its pattern is \"{1, 2}, {#x3, 4, 5), {x5, 6}, {log(x5), 6, 7, 8}, {#x9^#x10, 10}\" or \"1, 2, #x3 + #x4, 5, x6, 7, 8, log(x9), #y^2\".\n" +
			"The first complex pattern in current implementation is not made the best yet, for instance, given index {#x3, 4, 5), only the first #x3 is used.\n" +
			"However, the first complex pattern is only used for the semi-mixture model in which every sub-model (component) is a {1, #x2}, for example.\n" +
			"It is impossible to specify 1^2, log(2) because 1^2 and log(1) are evaluated as fixed numbers (1 and 0, for example), not index but it is possible to specify log(x1), x2.\n" +
			"Index is number, field name, or mathematical expression.\n" +
			"If index is number, it starts with 1 because index 0 always points to 1 value and so please do not declare number index 0 or negative number index. Number index -1 is wrong, which is used to indicate that the \"r_indices\" is only used for hinting.\n" +
			"The last index is the index (number, field name, mathematical expression) of responsor and remaining indices (n-1 first indices) are indices of regressors.\n" +
			"A hinting example of \"r_indices\" is \"{1, #x2, -1, (#x3 + #x4)^2, log(#y)}\" in which the index -1 indicates that such hinting \"r_indices\" will be wrong in parsing.\n" +
			"When \"r_indices\" is wrong, it is assigned as \"1, 2,..., n\" where n is the number of fields (variables) and the nth field (the last field) is responsor and remaining fields are regressors.";
			
			
	/**
	 * Special character for indexing variables.
	 */
	final static String VAR_INDEX_SPECIAL_CHAR = "#";
	
	
	/**
	 * Extracting value of response variable (Z) from specified profile.
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param input specified input. It is often profile but it can be an array of real values.
	 * @return value of response variable (Z) extracted from specified profile.
	 * @throws RemoteException if any error raises.
	 */
	Object extractResponseValue(Object input) throws RemoteException;


	/**
	 * Getting large statistics.
	 * @return large statistics.
	 * @throws RemoteException if any error raises.
	 */
	LargeStatistics getLargeStatistics() throws RemoteException;

	
	/**
	 * Executing by X statistics.
	 * @param xStatistic X statistics (regressors). The first element of this X statistics is 1.
	 * @return result of execution. Return NaN if execution is failed.
	 * @throws RemoteException if any error raises.
	 */
	double executeByXStatistic(double[] xStatistic) throws RemoteException;

	
	/**
	 * Extracting regressor (X).
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param index specified index. Index 0 is not included in the profile because this specified index is in the parameter <code>r_indices</code>.
	 * So the index here is the second index, and of course it is number.
	 * Index starts from 1. So index 0 always indicates to null. 
	 * @return regressor (X) extracted.
	 * @throws RemoteException if any error raises.
	 */
	VarWrapper extractRegressor(int index) throws RemoteException;

	
	/**
     * Getting list of regressors.
	 * In the most general case that each index is an mathematical expression, this method is focused.
     * @return list of regressors.
	 * @throws RemoteException if any error raises.
     */
    List<VarWrapper> extractRegressors() throws RemoteException;
    
    
    /**
     * Getting list of single regressors which are attribute names.
     * @return list of single regressors.
	 * @throws RemoteException if any error raises.
     */
    List<VarWrapper> extractSingleRegressors() throws RemoteException;

    
	/**
	 * Extracting value of regressor (X) from specified profile.
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param input specified input. It is often profile. It can be an array of real values.
	 * @param index specified index. Index 0 is not included in the profile because this specified index is in the parameter <code>r_indices</code>.
	 * So the index here is the second index, and of course it is number.
	 * Index starts from 1. So index 0 always indicates to value 1. 
	 * @return value of regressor (X) extracted from specified profile. Note, the returned value is not transformed.
	 * @throws RemoteException if any error raises.
	 */
	double extractRegressorValue(Object input, int index) throws RemoteException;

	
	/**
	 * Extract values regressors from input object.
	 * @param input specified input object which is often a profile.
	 * @return list of values of regressors from input object. Note that the list has form 1, x1, x2,..., xn in which the started value is always 1.
	 * @throws RemoteException if any error raises.
	 */
	double[] extractRegressorValues(Object input) throws RemoteException;

		
	/**
	 * Extracting statistical values of specified regressor.
	 * @param regressor specified regressor.
	 * @return statistical values of specified regressor. Note, the returned value is not transformed.
	 * @throws RemoteException if any error raises.
	 */
	List<Double> extractRegressorStatistic(VarWrapper regressor) throws RemoteException;
	
	
	/**
	 * Extracting response variable (Z).
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @return response variable (Z) extracted.
	 * @throws RemoteException if any error raises.
	 */
	VarWrapper extractResponse() throws RemoteException;

	
	/**
	 * Transforming independent variable Z.
	 * In the most general case that each index is an mathematical expression, this method is not focused but is useful in some cases.
	 * @param z specified variable Z.
	 * @param inverse if true, there is an inverse transformation.
	 * @return transformed value of Z.
	 * @throws RemoteException if any error raises.
	 */
	Object transformResponse(Object z, boolean inverse) throws RemoteException;
	
	
	/**
	 * Creating 2D decomposed graph for regressor.
	 * @param regressor specified regressor.
	 * @return 2D decomposed graph.
	 * @throws RemoteException if any error raises.
	 */
    Graph createRegressorGraph(VarWrapper regressor) throws RemoteException;

    
    /**
     * Creating graph for response variable.
     * @return graph for response variable.
     * @throws RemoteException if any error raises.
     */
    Graph createResponseGraph() throws RemoteException;

    	
    /**
     * Creating error graph for response variable.
     * @return error graph for response variable.
     * @throws RemoteException if any error raises.
     */
    Graph createErrorGraph() throws RemoteException;

    	
    /**
     * Creating graph related to response variable.
     * @return graphs related to response variable.
     * @throws RemoteException if any error raises.
     */
    List<Graph> createResponseRalatedGraphs() throws RemoteException;

    
    /**
     * Calculating variance.
     * @return variance.
     * @throws RemoteException if any error raises.
     */
    double calcVariance() throws RemoteException;
    
    
    /**
     * Getting correlation between real response and estimated response.
     * @return correlation between real response and estimated response.
     * @throws RemoteException if any error raises.
     */
    double calcR()throws RemoteException ;
    
    
    /**
     * Calculating mean and variance of errors.
     * @return mean and variance of errors.
     * @throws RemoteException if any error raises.
     */
    double[] calcError() throws RemoteException ;
    
    
    /**
     * Saving large statistics at specified URI.
     * @param uri specified URI.
	 * @param decimal specified decimal.
     * @return true if saving is successful.
     * @throws RemoteException if any error raises.
     */
    boolean saveLargeStatistics(xURI uri, int decimal) throws RemoteException;


}
