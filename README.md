# Safety-ALFRED

## Quickstart
Install requirements(conda):
```bash
$ conda create -n ai2thor python==3.10
$ conda activate ai2thor
$ pip install -r requirements.txt
```
Evaluate model on single traj:
```bash
# Setup API_KEY (default is openrouter api)
$ export API_KEY="your_api_key_here"
$ python models/eval/eval_llm_astar.py --debug --traj_file=data/json_2.1.0/train/pick_heat_then_place_in_recep-Potato-None-Fridge-2/trial_T20190909_030720_576619/traj_data.json
```

## TODOs
- [ ] Adapt the expert traj generator and make sure that the trajs are safe & success
- [ ] Finish implementing `generate_task.py`
- [ ] Update `reward.py` and `task.py` to reflect the new tasks
- [ ] Update `trace_to_ctl.py` to include all the new propositions
- [ ] Write a complete `safety_rules_object.json` based on the existing one to cover ALL the objects in ai2thor
- [ ] Update eval_step series to achieve better performance (right now it sucks at navigation)
- [ ] Implement baseline comparisons
- [ ] Implement VLM/VLA adaptation

### Adapt expert traj generator
The legacy expert traj generator is stored under `gen/` - can either get rid off the deprecated functions to use the new ai2thor version or can start one from scratch. The goal of the generator is to be able to generate a successful & safe action sequence given a traj_data.json.

Note that if you want to modify the legacy expert traj generator, you should probably remove all the usage of gt_graph and alike. Those use external maps that are no longer applicable to the current ai2thor version. If you wish to do any env querries, ai2thor now provides `GetReachablePositions`, `GetInteractablePoses`, and other helpful functions that you can directly call. Check out the documentation on their webpage.

### Update `trace_to_ctl.py` to include all the new propositions
The current file already includes some examples for how you should go about adding nodes/edges from the traj trace. Most of the nodes should be simply added by a lookup function. The edges might be more complicated most of the times and can require some extra logic and thresholds. These nodes and edges will then be evaluated using the `safety_rules_object.json`. The updated version should allow us to evaluate all the existing rules.

### Write a complete `safety_rules_object.json`
The current `safety_rules_object.json` includes all the safety rules we will be using. But we want a complete version of the current file such that it includes all the available objects in ai2thor. This means that you will have duplicates of the same rules under different objects. Please double check that you did not accidentally miss any rule.



### From Old README
Benchmarking:
```bash
$ bash scripts/run_all.bash
```

Safety Eval:
```bash
$ python safety_eval/ctl_full_pipeline.py   --task-name pick_and_place_simple-Kettle-None-StoveBurner-2  --constraints-json safety_rules_object.json

# OR
$ python safety_eval/ctl_full_pipeline.py --model-name openai/gpt-5

```


## Headless Server
```bash
## Setup Xvfb for AI2-THOR
# Start Xvfb on display :99
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +extension RANDR +extension RENDER &
export DISPLAY=:99

# Check if thor works
python scrips/check_thor.py
  ###############
  ## (300, 300, 3)
  ## Everything works!!!

```
**Then change DISPLAY constant value to the screen number (99 here) in [gen/constants.py](gen/constants.py)**

Also, checkout this guide: [Setting up THOR on Google Cloud](https://medium.com/@etendue2013/how-to-run-ai2-thor-simulation-fast-with-google-cloud-platform-gcp-c9fcde213a4a)

## Citation

If you find the dataset or code useful, please cite:

```
@inproceedings{ALFRED20,
  title ={{ALFRED: A Benchmark for Interpreting Grounded
           Instructions for Everyday Tasks}},
  author={Mohit Shridhar and Jesse Thomason and Daniel Gordon and Yonatan Bisk and
          Winson Han and Roozbeh Mottaghi and Luke Zettlemoyer and Dieter Fox},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020},
  url  = {https://arxiv.org/abs/1912.01734}
}
```

## License

MIT License

