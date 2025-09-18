import EvolutionManager 


def get_logbook():
    manager = EvolutionManager()
    best_ind, logbook = manager.run_evolution()
    # Option 1: use the return value
    print(logbook.select("avg"))
    # Option 2: use the field stored in the object
    print(manager.logbook.select("max"))

    return logbook

if __name__ == "__main__":
    get_logbook()



