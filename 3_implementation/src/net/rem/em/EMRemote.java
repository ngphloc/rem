/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.em;

import java.rmi.RemoteException;

import net.hudup.core.alg.ExecutableAlgRemote;

/**
 * This interface represents a remote EM algorithm.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface EMRemote extends EMRemoteTask, ExecutableAlgRemote {
	

	@Override
	Object getStatistics() throws RemoteException;


}
