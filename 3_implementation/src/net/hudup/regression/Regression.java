package net.hudup.regression;

import net.hudup.core.alg.TestingAlg;

/**
 * <code>Regression</code> is the most abstract interface for all regression algorithms.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Regression extends TestingAlg {

	
	/**
	 * Getting index of response variable (Z).
	 * @return response variable (Z).
	 */
	int getResponseIndex();
	
	
}
