import carla


def main():
    try:
        client = carla.Client('localhost', 2000)
        print('client: ', client, 'connection alive')

    finally:
        print('destroying actors')

if __name__ == '__main__':

    main()
