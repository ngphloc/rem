/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.em;

import net.hudup.core.alg.ExecutableAlg;

/**
 * <code>EM</code> is the most abstract interface for expectation maximization (EM) algorithm.
 * Its main method is learning method used to learn parameters.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface EM extends EMRemoteTask, ExecutableAlg {

	
	/**
	 * Maximum number of iterations.
	 */
	final static int EM_MAX_ITERATION = 1000;
	
	
	/**
	 * Default epsilon for terminated condition, which is the bias between current parameter and estimated parameter. 
	 */
	final static double EM_EPSILON = 0.001;
	
	
	/**
	 * Default value for epsilon ratio mode.
	 */
	final static boolean EM_EPSILON_RATIO_MODE = true;

	
	/**
	 * Getting current iteration.
	 * @return current iteration. Return 0 if the algorithm does not run yet or run failed. 
	 */
	int getCurrentIteration();
	
	
	/**
	 * Getting current parameter.
	 * @return current parameter. Return null if the algorithm does not run yet or run failed. 
	 */
	Object getCurrentParameter();
	
	
	/**
	 * Getting estimated parameter.
	 * @return estimated parameter. Return null if the algorithm does not run yet or run failed. 
	 */
	Object getEstimatedParameter();
	
	
}
