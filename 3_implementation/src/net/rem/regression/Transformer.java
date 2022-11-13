/**
 * REM: REGRESSION MODELS BASED ON EXPECTATION MAXIMIZATION ALGORITHM
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression;

import java.rmi.RemoteException;

/**
 * This interface presensent a transformer for regressors and response.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Transformer {

	
	/**
	 * Transforming independent variable X. In the most general case that each index is an mathematical expression, this method is not focused.
	 * @param x specified variable X.
	 * @param inverse if true, there is an inverse transformation.
	 * @return transformed value of X.
	 * @throws RemoteException if any error raises.
	 */
	Object transformRegressor(Object x, boolean inverse) throws RemoteException;


	/**
	 * Transforming independent variable Z. In the most general case that each index is an mathematical expression, this method is not focused but is useful in some cases.
	 * @param z specified variable Z.
	 * @param inverse if true, there is an inverse transformation.
	 * @return transformed value of Z.
	 * @throws RemoteException if any error raises.
	 */
	Object transformResponse(Object z, boolean inverse) throws RemoteException;
	
	
}
