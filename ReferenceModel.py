#!/usr/bin/env python
"""Generate a model for use with the progressive hedging algorithm.

If loaded as a module by runph, this creates a "model" object which can be
used to define the model.

If called as a script, this creates pha .dat files in the pha subdirectory
(e.g., inputs/pha/) with all the data needed to instantiate the model.

Note: The model will be built using the standard 'switch solve' configuration
based on options.txt, modules.txt and any command line arguments. The model
should include pha.py as one of its modules (e.g., in modules.txt).
"""

# turn on universal exception debugging (on the runph side)
# import debug

import os, shlex
import switch_mod.solve
    
# if imported by another module, just create the model (which will be extracted by the other module)
# if loaded directly with an output file argument, write the dat file

if __name__ == '__main__':
    # called directly from command line; save data and exit
    instance = switch_mod.solve.main(return_instance=True)
    if hasattr(instance, "save_pha_dat_files"):
        instance.save_pha_dat_files()
        # instance.save_pha_rho_file()   # this is now called from get_scenario_data.py
    else:
        raise RuntimeError(
            "The pha module is needed in order to save pha .dat files, but it is not included "
            "in the current model configuration. "
            "Please add the 'pha' module to the model via modules.txt or another configuration option."
        )
else:
    # The module was imported; create the model so it can be used by runph 
    print "defining model..."
    args = switch_mod.solve.get_option_file_args()
    # Take arguments from environment variables, since all command-line arguments 
    # are for runph at this point
    for k, v in os.environ.iteritems():
        if k.startswith('switch_'):
            args.append('--' + k[len('switch_'):].replace('_', '-'))
            if v.lower() == 'true' or v.lower() == '':
                # special handling for on/off flags
                pass
            else:
                args.extend(shlex.split(v)) # split at quotes just like argv
    model = switch_mod.solve.main(args=args, return_model=True)


