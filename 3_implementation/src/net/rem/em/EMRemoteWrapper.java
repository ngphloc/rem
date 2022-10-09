/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.em;

import java.rmi.RemoteException;

import net.hudup.core.alg.ExecutableAlgRemoteWrapper;
import net.hudup.core.data.DataConfig;
import net.hudup.core.logistic.BaseClass;
import net.hudup.core.logistic.LogUtil;

/**
 * The class is a wrapper of remote executable algorithm. This is a trick to use RMI object but not to break the defined programming architecture.
 * In fact, RMI mechanism has some troubles or it it affect negatively good architecture.
 * For usage, an algorithm as REM will has a pair: REM stub (remote executable algorithm) and REM wrapper (normal executable algorithm).
 * The server creates REM stub (remote executable algorithm) and the client creates and uses the REM wrapper as normal executable algorithm.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@BaseClass //The annotation is very important which prevent Firer to instantiate the wrapper without referred remote object. This wrapper is not normal algorithm.
public class EMRemoteWrapper extends ExecutableAlgRemoteWrapper implements EM, EMRemote {

	
	/**
	 * Default serial version UID.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with specified EM algorithm.
	 * @param remoteEM remote EM algorithm.
	 */
	public EMRemoteWrapper(EMRemote remoteEM) {
		super(remoteEM);
	}

	
	/**
	 * Constructor with specified EM algorithm and exclusive mode.
	 * @param remoteEM remote EM algorithm.
	 * @param exclusive exclusive mode.
	 */
	public EMRemoteWrapper(EMRemote remoteEM, boolean exclusive) {
		super(remoteEM, exclusive);
	}

	
	@Override
	public int getCurrentIteration() {
		if (remoteAlg instanceof EM)
			return ((EM)remoteAlg).getCurrentIteration();
		else {
			LogUtil.warn("getCurrentIteration() not supported");
			return -1;
		}
	}

	
	@Override
	public Object getCurrentParameter() {
		if (remoteAlg instanceof EM)
			return ((EM)remoteAlg).getCurrentParameter();
		else {
			LogUtil.warn("getCurrentParameter() not supported");
			return null;
		}
	}

	
	@Override
	public Object getEstimatedParameter() {
		if (remoteAlg instanceof EM)
			return ((EM)remoteAlg).getEstimatedParameter();
		else {
			LogUtil.warn("getEstimatedParameter() not supported");
			return null;
		}
	}

	
	@Override
	public Object getStatistics() throws RemoteException {
		return ((EM)remoteAlg).getStatistics();
	}

	
	@Override
	public String[] getBaseRemoteInterfaceNames() throws RemoteException {
		return new String[] {EMRemote.class.getName()};
	}

	
	@Override
	public DataConfig createDefaultConfig() {
		if (remoteAlg instanceof EM)
			return ((EM)remoteAlg).createDefaultConfig();
		else {
			LogUtil.warn("Wrapper of remote EM algorithm does not support createDefaultConfig()");
			return null;
		}
	}


}
