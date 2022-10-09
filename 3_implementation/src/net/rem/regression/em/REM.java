/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression.em;

import net.rem.em.EM;
import net.rem.regression.RM;

/**
 * This interface is an indicator of any algorithm that applying expectation maximization algorithm into learning regression model.
 * It is called REM algorithm.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface REM extends REMRemoteTask, RM, EM {

	
}
