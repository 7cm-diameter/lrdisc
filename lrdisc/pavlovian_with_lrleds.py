from amas.agent import Agent, NotWorkingError
from comprex.agent import ABEND, NEND, OBSERVER, RECORDER, START
from comprex.config import Experimental
from comprex.scheduler import TrialIterator2, repeat, unif_rng, elementwise_shuffle
from comprex.util import timestamp
from pino.ino import HIGH, LOW, Arduino


async def control(agent: Agent, ino: Arduino, expvars: Experimental) -> None:
    light_duration = expvars.get("light-duration", 1.)
    reward_duration = expvars.get("reward-duration", 0.02)

    light_pins = expvars.get("light-pin", [4, 8])
    reward_pin = expvars.get("reward-pin", [2, 3])

    mean_isi = expvars.get("inter-stimulus-interval", 19.)
    range_isi = expvars.get("interval-range", 10.)

    number_of_trial = expvars.get("number-of-trial", 200)
    isis = unif_rng(mean_isi, range_isi, number_of_trial)
    reward_positions = elementwise_shuffle(repeat([0, 1], [number_of_trial // 2, number_of_trial // 2]))
    trials = TrialIterator2(isis)

    try:
        while agent.working():
            agent.send_to(RECORDER, timestamp(START))
            for i, isi in trials:
                print(f"Trial {i}: Cue will be presented {isi} secs after.")
                await agent.sleep(isis[i])
                agent.send_to(RECORDER, timestamp(light_pins[0]))
                agent.send_to(RECORDER, timestamp(light_pins[-1]))
                ino.digital_write(light_pins[0], HIGH)
                ino.digital_write(light_pins[-1], HIGH)
                await agent.sleep(light_duration)
                agent.send_to(RECORDER, timestamp(-light_pins[0]))
                agent.send_to(RECORDER, timestamp(-light_pins[-1]))
                ino.digital_write(light_pins[0], LOW)
                ino.digital_write(light_pins[-1], LOW)
                agent.send_to(RECORDER, timestamp(reward_pin))
                ino.digital_write(reward_pin, HIGH)
                await agent.sleep(reward_duration)
                agent.send_to(RECORDER, timestamp(-reward_pin))
                ino.digital_write(reward_pin, LOW)
            agent.send_to(OBSERVER, NEND)
            agent.send_to(RECORDER, timestamp(NEND))
            agent.finish()
    except NotWorkingError:
        agent.send_to(OBSERVER, ABEND)
        agent.send_to(RECORDER, timestamp(ABEND))
        agent.finish()
    return None


if __name__ == '__main__':
    from os import mkdir
    from os.path import exists, join

    from amas.connection import Register
    from amas.env import Environment
    from comprex.agent import Observer, Reader, Recorder, _self_terminate
    from comprex.config import PinoClap
    from comprex.util import get_current_file_abspath, namefile
    from pino.ino import Arduino, Comport

    config = PinoClap().config

    com = Comport() \
        .apply_settings(config.comport) \
        .set_timeout(1.0) \
        .deploy() \
        .connect()

    ino = Arduino(com)
    ino.apply_pinmode_settings(config.pinmode)

    data_dir = join(get_current_file_abspath(__file__), "data")
    if not exists(data_dir):
        mkdir(data_dir)
    filename = join(data_dir, namefile(config.metadata))

    controller = Agent("Controller") \
        .assign_task(control, ino=ino, expvars=config.experimental) \
        .assign_task(_self_terminate)

    # Use built-in agents
    reader = Reader(ino=ino)
    recorder = Recorder(filename=filename)
    observer = Observer()

    agents = [controller, reader, recorder, observer]
    register = Register(agents)
    env = Environment(agents)

    try:
        env.run()
    except KeyboardInterrupt:
        observer.send_all(ABEND)
        observer.finish()
