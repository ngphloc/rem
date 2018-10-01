package net.hudup.regression;

import net.hudup.core.alg.TestingAlg;

/**
 * <code>Regression</code> is the most abstract interface for all regression algorithms.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface RM extends TestingAlg {
	
	
	/**
	 * Name of regression indices field.
	 */
	public final static String R_INDICES_FIELD = "r_indices";

	
	/**
	 * Default regression indices field.
	 */
	public final static String R_INDICES_DEFAULT = "{1, #x2, -1, (#x3 + #x4)^2, log(#y)}"; //Use default indices in which n-1 first variables are regressors and the last variable is response variable
	
	
	/**
	 * Special character for indexing variables.
	 */
	public final static String VAR_INDEX_SPECIAL_CHAR = "#";
	
	
	/**
	 * Extracting value of response variable (Z) from specified profile.
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param input specified input. It is often profile.
	 * @return value of response variable (Z) extracted from specified profile.
	 */
	Object extractResponseValue(Object input);
	
	
}
