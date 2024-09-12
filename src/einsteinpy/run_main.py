
import logging
import grfuncs



formater = logging.Formatter('%(asctime)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formater)

logger = logging.getLogger("simplelog")
logger.addHandler(handler)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logger.info('setting up metric')
    #coords, metric = grfuncs.setup_schwarzschild()
    #coords, metric = grfuncs.setup_antidesitter()
    coords, metric = grfuncs.setup_sphsym()
    #coords, metric = grfuncs.setup_5D_flat()

    logger.info('calculating curvature')
    curv = grfuncs.calc_curvature(coords, metric, logger)
    logger.info('finished')

    print(curv[0])