from time import perf_counter
from amas.agent import Agent, NotWorkingError
from comprex.agent import ABEND, NEND, OBSERVER, RECORDER, START
from comprex.config import Experimental
from comprex.scheduler import TrialIterator2, repeat, unif_rng, elementwise_shuffle
from comprex.util import timestamp
from pino.ino import HIGH, LOW, Arduino


CONTROLLER = "Contoroller"


async def flush_message_for(agent: Agent, duration: float):
    while duration >= 0. and agent.working():
        s = perf_counter()
        await agent.try_recv(duration)
        e = perf_counter()
        duration -= e - s


async def fixed_interval_with_postpone(agent: Agent, duration: float, correct: str, limit: float = 3.):
    duration_count_down = duration
    while duration_count_down >= 0. and agent.working():
        mail = await agent.try_recv(limit)
        duration_count_down -= perf_counter()
        if mail is None:
            break
        _, response = mail
        if response != correct:
            duration_count_down = duration


async def control(agent: Agent, ino: Arduino, expvars: Experimental) -> None:
    light_duration = expvars.get("light-duration", 1.)
    reward_duration = expvars.get("reward-duration", 0.02)

    light_pins = expvars.get("light-pin", [4, 8])
    reward_pins = expvars.get("reward-pin", [2, 3])
    response_pins = list(map(str, expvars.get("response-pin", [-9, -10])))

    mean_isi = expvars.get("inter-stimulus-interval", 19.)
    range_isi = expvars.get("interval-range", 10.)

    number_of_trial = expvars.get("number-of-trial", 200)
    isis = unif_rng(mean_isi, range_isi, number_of_trial)
    corrects = elementwise_shuffle(repeat(reward_pins, [number_of_trial // 2, number_of_trial // 2]))
    trials = TrialIterator2(isis, corrects)

    try:
        while agent.working():
            agent.send_to(RECORDER, timestamp(START))
            for i, isi, correct in trials:
                print(f"Trial {i}: Cue will be presented {isi} secs after.")
                await flush_message_for(agent, isi)
                agent.send_to(RECORDER, timestamp(light_pins[correct]))
                ino.digital_write(light_pins[correct], HIGH)
                await fixed_interval_with_postpone(agent, light_duration, response_pins[correct])
                agent.send_to(RECORDER, timestamp(-light_pins[correct]))
                ino.digital_write(light_pins[correct], LOW)
                ino.digital_write(reward_pins[correct], HIGH)
                await agent.sleep(reward_duration)
                agent.send_to(RECORDER, timestamp(-reward_pins[correct]))
            agent.send_to(OBSERVER, NEND)
            agent.send_to(RECORDER, timestamp(NEND))
            agent.finish()
    except NotWorkingError:
        agent.send_to(OBSERVER, ABEND)
        agent.send_to(RECORDER, timestamp(ABEND))
        agent.finish()
    return None


async def read(agent: Agent, ino: Arduino, expvars: Experimental):
    response_pins = list(map(str, expvars.get("response-pin", [-9, -10])))
    try:
        while agent.working():
            input_: bytes = await agent.call_async(ino.read_until_eol)
            if input_ is None:
                continue
            parsed_input = input_.rstrip().decode("utf-8")
            agent.send_to(RECORDER, timestamp(parsed_input))
            if parsed_input in response_pins:
                agent.send_to(CONTROLLER, parsed_input)
    except NotWorkingError:
        ino.cancel_read()


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

    controller = Agent(CONTROLLER) \
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
