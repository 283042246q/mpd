try:
    from .infer_dynamic_server_marvin_bimanual import main
except ImportError:
    from infer_dynamic_server_marvin_bimanual import main


if __name__ == "__main__":
    main()
