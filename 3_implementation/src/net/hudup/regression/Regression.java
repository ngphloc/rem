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
	 * Extracting value of response variable (Z) from specified profile.
	 * @param profile specified profile.
	 * @return value of response variable (Z) extracted from specified profile.
	 */
	double extractResponse(Profile profile);
	
	
}
