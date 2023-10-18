from time import perf_counter
from amas.agent import Agent, NotWorkingError
from comprex.agent import ABEND, NEND, OBSERVER, RECORDER, START
from comprex.config import Experimental
from comprex.scheduler import TrialIterator2, repeat, unif_rng, elementwise_shuffle, geom_rng
from comprex.util import timestamp
from pino.ino import HIGH, LOW, Arduino


CONTROLLER = "Contoroller"


async def flush_message_for(agent: Agent, duration: float):
    while duration >= 0. and agent.working():
        s = perf_counter()
        await agent.try_recv(duration)
        e = perf_counter()
        duration -= e - s


async def control(agent: Agent, ino: Arduino, expvars: Experimental):
    reward_duration = expvars.get("reward-duration", 0.02)
    number_of_block = expvars.get("numbe-of-block", 20)
    mean_block_length = expvars.get("block-length", 15) + .5
    block_range = expvars.get("block-range", 5)

    mean_ici = expvars.get("mean-ici", 300.)
    range_ici = expvars.get("range-ici", 120.)

    mean_required_response = expvars.get("mean_required-response", 10)
    range_required_response = expvars.get("range-required-response", 5)
    timelimit = expvars.get("timelimit", 30.)

    light_pins = expvars.get("light-pin", [4, 8])
    reward_pins = expvars.get("reward-pin", [2, 3])
    response_pins = list(map(str, expvars.get("response-pin", [-9, -10])))

    async def variable_ratio_with_limit(block_lenght: int, component: int, current_component: int):
        ratio = list(map(int, unif_rng(mean_required_response + .5, range_required_response, block_lenght)))
        agent.send_to(RECORDER, timestamp(light_pins[component]))
        ino.digital_write(light_pins[component], HIGH)
        i = 0
        for r in ratio:
            i += 1
            print(f"{i}/{block_lenght} trial in {current_component} component")
            count = 0
            while count < r:
                mail = await agent.try_recv(timelimit)
                if mail is None:
                    break
                _, response = mail
                if response == response_pins[component]:
                    count += 1
                else:
                    if count > 0:
                        count -= 1
            agent.send_to(RECORDER, timestamp(reward_pins[component]))
            ino.digital_write(reward_pins[component], HIGH)
            await agent.sleep(reward_duration)
            agent.send_to(RECORDER, timestamp(-reward_pins[component]))
            ino.digital_write(reward_pins[component], LOW)
        agent.send_to(RECORDER, timestamp(-light_pins[component]))
        ino.digital_write(light_pins[component], LOW)

    block_length = list(map(int, unif_rng(mean_block_length, block_range, number_of_block)))
    icis = unif_rng(mean_ici, range_ici, number_of_block)
    components = [i % 2 for i in range(number_of_block)]

    trials = TrialIterator2(block_length, icis, components)

    try:
        while agent.working():
            for current, block, ici, component in trials:
                await variable_ratio_with_limit(block, component, current)
                await flush_message_for(agent, ici)
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
    from comprex.agent import Observer, Recorder, _self_terminate, READER
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
    reader = Agent(READER) \
        .assign_task(read, ino=ino, expvars=config.experimental) \
        .assign_task(_self_terminate)
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
