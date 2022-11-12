/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression.ui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;

import net.hudup.core.logistic.ClipboardUtil;
import net.hudup.core.logistic.LogUtil;
import net.hudup.core.logistic.ui.TextArea;
import net.hudup.core.logistic.ui.UIUtil;
import net.rem.regression.RM;

/**
 * This class is the text area to show basic infomation of a regression model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class RMTextArea extends TextArea {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Internal regression model.
	 */
	protected RM rm = null;
	
	
	/**
	 * Constructor with specified regression model.
	 * @param rm specified regression model.
	 */
	public RMTextArea(RM rm) {
		super();
		this.rm = rm;
		
		setEditable(false);
		setText(getRMDescription());
		setRows(3);
		setAutoscrolls(true);
		setCaretPosition(0);
		
		setToolTipText(getRMDescription());
	}
	
	
	@Override
	protected JPopupMenu createContextMenu() {
		if (rm == null) return null;
		
		JPopupMenu contextMenu = new JPopupMenu();
		
		JMenuItem miCopyDesc = UIUtil.makeMenuItem(null, "Copy", 
			new ActionListener() {
				
				public void actionPerformed(ActionEvent e) {
					ClipboardUtil.util.setText(getRMDescription());
				}
			});
		contextMenu.add(miCopyDesc);

		JMenuItem miZoom = UIUtil.makeMenuItem(null, "Zoom", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					zoom();
				}
			});
		contextMenu.add(miZoom);

		return contextMenu;
	}
	
	
	/**
	 * Getting description of regression model.
	 * @return description of regression model.
	 */
	private String getRMDescription() {
		if (rm == null)
			return "";
		
		try {
			return rm.getDescription();
		} catch (Exception e) {
			LogUtil.trace(e);
		}
		
		return "";
	}
	
	
}
