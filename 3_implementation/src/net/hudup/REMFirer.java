/**
 * AI project provide artificial intelligence solutions
 * (C) Copyright by Loc Nguyen
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.hudup;

import net.hudup.core.Firer;
import net.hudup.core.alg.AlgRemote;
import net.hudup.core.alg.AlgRemoteWrapper;
import net.rem.em.EMRemote;
import net.rem.em.EMRemoteWrapper;
import net.rem.regression.RMRemote;
import net.rem.regression.RMRemoteWrapper;

/**
 * This is advanced plug-in manager which derives from {@link Firer}.
 * 
 * @author Loc Nguyen
 * @version 2.0
 *
 */
public class REMFirer extends Firer {

	
	@Override
	public void fireSimply() {
		super.fireSimply();
	}

	
	@Override
	public AlgRemoteWrapper wrap(AlgRemote remoteAlg, boolean exclusive) {
		if (remoteAlg instanceof RMRemote)
			return new RMRemoteWrapper((RMRemote)remoteAlg, exclusive);
		else if (remoteAlg instanceof EMRemote)
			return new EMRemoteWrapper((EMRemote)remoteAlg, exclusive);
		else
			return super.wrap(remoteAlg, exclusive);
	}

	
}
