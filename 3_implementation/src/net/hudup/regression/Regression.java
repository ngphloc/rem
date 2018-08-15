package net.hudup.regression;

import net.hudup.core.alg.TestingAlg;
import net.hudup.core.data.Profile;

/**
 * <code>Regression</code> is the most abstract interface for all regression algorithms.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Regression extends TestingAlg {
	
	
	/**
	 * Name of regression indices field.
	 */
	public final static String R_INDICES_FIELD = "r_indices";

	
	/**
	 * Default regression indices field.
	 */
	public final static String R_INDICES_FIELD_DEFAULT = "{0, #x1, -1, (#x2 + #x3)^2, log(#y)}"; //Use default indices in which n-1 first variables are regressors and the last variable is response variable
	
	
	/**
	 * Special character for indexing variables.
	 */
	public final static String VAR_INDEX_SPECIAL_CHAR = "#";
	
	
	/**
	 * Extracting value of response variable (Z) from specified profile.
	 * @param profile specified profile.
	 * @return value of response variable (Z) extracted from specified profile.
	 */
	Object extractResponse(Profile profile);
	
	
}
