package net.hudup.regression.em.ui;

import java.awt.Component;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPopupMenu;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;

import net.hudup.core.logistic.ClipboardUtil;
import net.hudup.core.logistic.ui.UIUtil;
import net.hudup.regression.em.REMImpl;

/**
 * This class is the text area to show basic infomation of a regression model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class REMTextArea extends JTextArea {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Internal regression model.
	 */
	protected REMImpl rem = null;
	
	
	/**
	 * Constructor with specified regression model.
	 * @param rem specified regression model.
	 */
	public REMTextArea(REMImpl rem) {
		this.rem = rem;
		
		setText(rem.getDescription());
		setEditable(false);
		setWrapStyleWord(true);
		setLineWrap(true);
		setRows(1);
		setAutoscrolls(true);
		setCaretPosition(0);
		
		addMouseListener(new MouseAdapter() {

			@Override
			public void mouseClicked(MouseEvent e) {
				// TODO Auto-generated method stub
				if(SwingUtilities.isRightMouseButton(e) ) {
					JPopupMenu contextMenu = createContextMenu();
					if(contextMenu != null) 
						contextMenu.show((Component)e.getSource(), e.getX(), e.getY());
				}
				else {
				}
			}
			
		});
		
		setToolTipText(rem.getDescription());
	}
	
	
	/**
	 * Create context menu.
	 * @return context menu.
	 */
	private JPopupMenu createContextMenu() {
		JPopupMenu contextMenu = new JPopupMenu();
		
		JMenuItem miCopyDesc = UIUtil.makeMenuItem(null, "Copy", 
			new ActionListener() {
				
				public void actionPerformed(ActionEvent e) {
					copyDescToClipboard();
				}
			});
		contextMenu.add(miCopyDesc);

		contextMenu.addSeparator();
		
		return contextMenu;
	}
	
	
	/**
	 * Copying regression description into clip board.
	 */
	private void copyDescToClipboard() {
		if (rem == null) {
			JOptionPane.showMessageDialog(
					this, 
					"Null regression model", 
					"Null regression model", 
					JOptionPane.ERROR_MESSAGE);
		}
		else
			ClipboardUtil.util.setText(rem.getDescription());
	}
	

}
